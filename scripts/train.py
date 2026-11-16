# -*- coding: utf-8 -*-
"""星辰 · 预训练脚本

用法：
  python scripts/train.py --config configs/tiny.yaml
  python scripts/train.py --config configs/tiny.yaml --max_iters 60 --device cuda
  python scripts/train.py --resume checkpoints/latest.pt

流程对应 DeepSeek LLM 报告的预训练管线：随机窗口采样 → 主干 CE 损失
（可选 + MTP 辅助损失）→ AdamW + warmup/cosine → 每步按负载更新 MoE 路由偏置。
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from xingchen import XingChen, load_config, get_batch, load_bin, param_summary, generate
from xingchen.tokenizer import ByteBPETokenizer


def lr_at(it: int, cfg) -> float:
    if it < cfg.train.warmup_iters:
        return cfg.train.lr * (it + 1) / cfg.train.warmup_iters
    t = (it - cfg.train.warmup_iters) / max(1, cfg.train.max_iters - cfg.train.warmup_iters)
    t = min(1.0, t)
    return cfg.train.min_lr + 0.5 * (cfg.train.lr - cfg.train.min_lr) * (1 + math.cos(math.pi * t))


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """分块 cross_entropy：把 [B*T, V] 的 fp32 临时张量切成若干块算，峰值显存线性下降。

    为什么要分块：vocab 16384、B*T=1024 时一次性 `.float()` 会生成 1024×16384 的
    fp32 张量（约 67 MB），是 8GB 卡上 micro-batch 上不去的直接原因之一。
    数学上等价：每块 reduction="sum"，最后除以 token 数，与一次性 mean 相同。
    精度上等价：每块仍用 fp32 计算，没有把 bf16 直接喂给 log_softmax。
    """
    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    n = flat_targets.numel()
    if chunk <= 0 or n <= chunk:
        return F.cross_entropy(flat_logits.float(), flat_targets)
    total = flat_logits.new_zeros(())          # 标量累加，保留 fp32
    for start in range(0, n, chunk):
        piece = flat_logits[start:start + chunk].float()
        total = total + F.cross_entropy(piece, flat_targets[start:start + chunk],
                                        reduction="sum")
    return total / n


def _file_sha(path: Path) -> str:
    """文件 SHA256。用来防止"换了分词器还拿旧 ckpt 续训"这种静默灾难。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_text(model, tok, cfg, device) -> str:
    ids = tok.encode("星辰", add_bos=True)
    out = generate(model, ids, max_new_tokens=80, temperature=0.8,
                   top_k=50, top_p=0.95, repetition_penalty=1.1,
                   eos_id=tok.eos_id)
    return tok.decode(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny.yaml"))
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--train_bin", default=None, help="覆盖配置里的训练集路径（记忆特训用）")
    ap.add_argument("--val_bin", default=None, help="覆盖配置里的验证集路径")
    ap.add_argument("--lr", type=float, default=None, help="覆盖配置里的学习率")
    ap.add_argument("--out_dir", default=None, help="覆盖检查点输出目录")
    ap.add_argument("--replay_bin", default=None,
                    help="replay 语料（一般指回配置里的预训练 train.bin）：按概率掺入，防止记忆特训把语言能力练坏")
    ap.add_argument("--replay_prob", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.train_bin:
        cfg.data.train_bin = args.train_bin
    if args.val_bin:
        cfg.data.val_bin = args.val_bin
    if args.lr is not None:
        cfg.train.lr = args.lr
        cfg.train.min_lr = args.lr * 0.1
    if args.out_dir:
        cfg.train.out_dir = args.out_dir

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---------------- 数据
    tok_path = ROOT / cfg.data.tokenizer_path
    tok = ByteBPETokenizer.load(tok_path)
    if tok.vocab_size != cfg.model.vocab_size:
        raise ValueError(f"配置 vocab_size={cfg.model.vocab_size} 与分词器 {tok.vocab_size} 不一致，"
                         f"请重跑 scripts/prepare_data.py")
    tok_sha = _file_sha(tok_path)

    # 多源按权重混合（优先）；否则退回单源 train_bin
    mix_arrays: list = []
    mix_cum: list = []
    train_data = None
    src_cfg = list(getattr(cfg.data, "sources", None) or [])
    if src_cfg:
        weights = []
        for item in src_cfg:
            p = ROOT / item["path"]
            if not p.exists():
                raise FileNotFoundError(f"数据源不存在：{p}")
            w = float(item.get("weight", 1.0))
            if w <= 0:
                raise ValueError(f"数据源 {p} 的 weight 必须 > 0")
            mix_arrays.append(load_bin(p))
            weights.append(w)
        total_w = sum(weights)
        cum, acc = [], 0.0
        for w in weights:
            acc += w / total_w
            cum.append(acc)
        cum[-1] = 1.0                       # 消除浮点误差，保证 random() 一定命中
        mix_cum = cum

        total_tokens = (cfg.train.max_iters * cfg.train.seq_len
                        * cfg.train.batch_size * cfg.train.grad_accum)
        print(f"[train] 混合 {len(mix_arrays)} 个源，合计 {sum(len(a) for a in mix_arrays):,} token"
              f"；本次计划消费 {total_tokens/1e9:.3f}B token")
        for item, arr, w in zip(src_cfg, mix_arrays, weights):
            ep = total_tokens * (w / total_w) / max(1, len(arr))
            tag = "  ⚠️ 重复过多，建议降权或补量" if ep > 4 else ""
            print(f"        {item['path']:<44} {len(arr)/1e6:>8.1f}M  "
                  f"权重 {w/total_w:5.1%}  预计 {ep:.2f} epoch{tag}")
    else:
        train_data = load_bin(ROOT / cfg.data.train_bin)

    val_data = load_bin(ROOT / cfg.data.val_bin)
    if mix_arrays:
        print(f"[train] 验证 token {len(val_data):,}（来自 {cfg.data.val_bin}）")
    else:
        print(f"[train] 训练 token {len(train_data):,} / 验证 token {len(val_data):,}")

    # 分域验证集（可选）：多源训练下只报一个总 loss 看不出哪一层在退化
    val_sets: list = []
    val_cfg = list(getattr(cfg.data, "val_sources", None) or [])
    if val_cfg:
        for item in val_cfg:
            p = ROOT / item["path"]
            if not p.exists():
                raise FileNotFoundError(f"验证源不存在：{p}")
            name = item.get("name") or p.stem
            arr = load_bin(p)
            if len(arr) < cfg.train.seq_len + 1:
                raise ValueError(f"验证源 {p} 只有 {len(arr):,} token，不足一个 seq_len")
            val_sets.append((name, arr))
        print(f"[train] 分域验证 {len(val_sets)} 个："
              + "、".join(f"{n}({len(a)/1e6:.1f}M)" for n, a in val_sets))
    else:
        val_sets.append((Path(cfg.data.val_bin).stem, val_data))

    def pick_train_source():
        """按权重抽一个源。用 bisect + random() 而不是 random.choices，单次约 0.4µs。"""
        i = bisect.bisect_right(mix_cum, random.random())
        return mix_arrays[min(i, len(mix_arrays) - 1)]

    replay_data = None
    if args.replay_bin:
        if mix_arrays:
            print("[train] 注意：已配置 data.sources（多源混合），--replay_bin 被忽略")
        else:
            rp = Path(args.replay_bin)
            if rp.exists():
                replay_data = load_bin(rp)
                print(f"[train] replay：{len(replay_data):,} token，每 batch {args.replay_prob:.0%} 概率掺入")
            else:
                print(f"[train] 警告：找不到 replay 语料 {rp}，本次不掺")

    # ---------------- 模型与优化器
    model = XingChen(cfg.model).to(device)
    model.grad_checkpoint = bool(getattr(cfg.train, "grad_checkpoint", False))
    stats = param_summary(model)
    print(f"[train] 模型 {cfg.name}：{stats['total']/1e6:.2f}M 参数 "
          f"(embedding {stats['embedding']/1e6:.2f}M, attention {stats['attention']/1e6:.2f}M, "
          f"moe/ffn {stats['moe/ffn']/1e6:.2f}M, 其他 {stats['mhc/mtp/其他']/1e6:.2f}M)")
    if model.grad_checkpoint:
        print(f"[train] 梯度检查点：开（seq_len={cfg.train.seq_len}，用算力换显存）")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=(cfg.train.beta1, cfg.train.beta2))
    fused_ok = device.type == "cuda"
    if fused_ok:
        try:
            opt = torch.optim.AdamW(
                [{"params": decay, "weight_decay": cfg.train.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=cfg.train.lr, betas=(cfg.train.beta1, cfg.train.beta2), fused=True)
        except Exception:
            pass

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_sha = ckpt.get("tokenizer_sha")
        if old_sha and old_sha != tok_sha:
            # 词表大小一致但内容不同时，load_state_dict 不会报错——id 的语义已经变了，
            # 训练会继续"正常"进行，loss 也能降，但学到的是错位映射，几周后才发现。
            raise ValueError(
                f"分词器与 checkpoint 不匹配，拒绝续训：\n"
                f"  ckpt  : {ckpt.get('tokenizer_path')}\n"
                f"          sha256={old_sha[:16]}\n"
                f"  当前  : {tok_path}\n"
                f"          sha256={tok_sha[:16]}\n"
                f"换过分词器就必须从头训；要接着跑原实验，请把 tokenizer_path 指回原来那份。")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_iter = ckpt["iter"] + 1
        torch.set_rng_state(ckpt["rng_torch"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
        print(f"[train] 从 {args.resume} 恢复，起始 iter {start_iter}")

    use_amp = device.type == "cuda" and cfg.train.amp == "bf16"
    out_dir = ROOT / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    B, T = cfg.train.batch_size, cfg.train.seq_len

    # 训练过程记录：每次刷新覆盖重写，供 scripts/monitor.py 面板读取
    log_path = out_dir / "train_log.jsonl"

    def jlog(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # resume 时**不要**清空日志：那会把此前几十小时的训练曲线一起删掉。
    # 只在从零开始时覆盖，续训接着追加（monitor.py 按 iter 读，重复的 iter 不会出现）。
    if start_iter == 0:
        open(log_path, "w", encoding="utf-8").close()
    jlog({"type": "meta", "name": cfg.name, "params_m": round(stats["total"] / 1e6, 2),
          "vocab": cfg.model.vocab_size, "seq": T, "batch": B,
          "max_iters": cfg.train.max_iters, "device": str(device)})

    def eval_val() -> dict:
        """分域验证，返回 {域名字: loss}。

        验证 loss 也走 ce_loss（分块），而不是原来那句一次性
        `cross_entropy(logits.view(-1, V).float())`——两者数学等价（verify_opt 已逐位验证），
        但分块版峰值显存低得多，能避免"训练刚好不 OOM、一 eval 就 OOM"。
        多源时 eval 成本按源数线性上升，eval_iters 是**每源各跑**这么多次。
        """
        model.eval()
        # 固定验证窗口。原先 eval 用随机窗口，"模型变好"和"这批窗口恰好更难"会混在
        # 一起，val loss 抖 0.02 时根本分不清是进步还是噪声。固定种子后每次 eval
        # 抽到同一批窗口，跨机也能复现。get_batch 用的是 CPU 侧 torch.randint，
        # 所以只需保存/恢复 CPU 的 RNG 状态。
        rng_state = torch.get_rng_state()
        torch.manual_seed(cfg.train.seed * 31 + 7)
        out = {}
        try:
            with torch.no_grad():
                for name, arr in val_sets:
                    losses = []
                    for _ in range(cfg.train.eval_iters):
                        x, y = get_batch(arr, B, T, device)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            logits, _ = model(x)
                            losses.append(ce_loss(logits, y, chunk=cfg.train.ce_chunk).item())
                    out[name] = sum(losses) / len(losses)
        finally:
            torch.set_rng_state(rng_state)
            model.train()
        return out

    # ---------------- 训练循环
    model.train()
    t0 = time.time()
    tokens_seen = 0
    for it in range(start_iter, cfg.train.max_iters):
        lr = lr_at(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        # 用 0-dim 张量累加 loss，整步只在最后同步一次。
        # 原来每个 micro-step 都调 .item()（grad_accum=8 就是 8 次），每次都会
        # 等 GPU 队列排空，等于把 micro-step 之间的流水线切断——这是纯浪费。
        loss_main_sum = None
        loss_mtp_sum = None
        for _ in range(cfg.train.grad_accum):
            if mix_arrays:
                src = pick_train_source()
            elif replay_data is not None and random.random() < args.replay_prob:
                src = replay_data          # 掺一条预训练批次，保住通用语言能力
            else:
                src = train_data
            x, y = get_batch(src, B, T, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, h = model(x)
                # y 已是右移一位的标签：logits[t] 直接对 y[t]
                loss_main = ce_loss(logits, y, chunk=cfg.train.ce_chunk)
                loss = loss_main
                if model.mtp is not None and cfg.model.mtp.loss_weight > 0:
                    # MTP：h[t] 与 x[t+1] 配对，预测 x[t+2] = y[t+1]
                    m_logits = model.mtp_forward(h[:, :-1], x[:, 1:])
                    loss_mtp = ce_loss(m_logits, y[:, 1:], chunk=cfg.train.ce_chunk)
                    loss = loss + cfg.model.mtp.loss_weight * loss_mtp
                    loss_mtp_sum = (loss_mtp.detach() if loss_mtp_sum is None
                                    else loss_mtp_sum + loss_mtp.detach())
            (loss / cfg.train.grad_accum).backward()
            loss_main_sum = (loss_main.detach() if loss_main_sum is None
                             else loss_main_sum + loss_main.detach())
            tokens_seen += x.numel()

        loss_main_acc = loss_main_sum.item()          # 全步唯一一次同步
        loss_mtp_acc = loss_mtp_sum.item() if loss_mtp_sum is not None else 0.0

        # NaN/Inf 立即停：8 天训练若在第 3 天发散，继续跑只是浪费电，
        # 而且会一路把正常 ckpt 覆盖成 NaN 权重。留一份现场供事后查。
        if not (math.isfinite(loss_main_acc) and math.isfinite(loss_mtp_acc)):
            nan_path = out_dir / "nan.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": cfg.to_dict(), "loss": loss_main_acc,
                        "rng_torch": torch.get_rng_state()}, nan_path)
            print(f"[train] ✗ iter {it} loss={loss_main_acc} / mtp={loss_mtp_acc} 出现 NaN/Inf，"
                  f"现场已存到 {nan_path.name}，训练中止。")
            jlog({"type": "nan", "iter": it + 1, "loss": loss_main_acc})
            return

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        max_dev = model.update_expert_bias(cfg.model.moe.balance_gamma) if model.moe_layers else None

        if (it + 1) % cfg.train.log_interval == 0 or it == start_iter:
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-6)
            mtp_str = f" | mtp {loss_mtp_acc / cfg.train.grad_accum:.3f}" if loss_mtp_acc else ""
            bal_str = f" | 负载偏差 {max_dev:.3f}" if max_dev is not None else ""
            mem = ""
            if device.type == "cuda":
                cur = torch.cuda.memory_allocated() / 2**30
                peak = torch.cuda.max_memory_allocated() / 2**30
                mem = f" | 显存 {cur:.2f}/{peak:.2f}GB"      # 当前/本区间峰值
            print(f"[train] iter {it:6d} | loss {loss_main_acc/cfg.train.grad_accum:.4f}{mtp_str}"
                  f" | lr {lr:.2e} | {tps/1000:.1f}k tok/s{bal_str}{mem}")
            jlog({"type": "train", "iter": it + 1,
                  "loss": round(loss_main_acc / cfg.train.grad_accum, 4),
                  "mtp": round(loss_mtp_acc / cfg.train.grad_accum, 4) if loss_mtp_acc else None,
                  "lr": lr, "tok_s": round(tps),
                  "load_dev": round(max_dev, 4) if max_dev is not None else None,
                  "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None})
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()   # 让下一区间重新统计，不显示历史最高
            t0 = time.time()
            tokens_seen = 0

        if (it + 1) % cfg.train.eval_interval == 0:
            vls = eval_val()
            head_name = val_sets[0][0]
            head = vls[head_name]
            # 多源时把其余域接在后面，一行看完；单源时 extra 为空，输出与旧版一致
            extra = "".join(f" | {n} {v:.4f}" for n, v in vls.items() if n != head_name)
            print(f"[train] iter {it:6d} | 验证 loss {head:.4f}（{head_name}）{extra}")
            jlog({"type": "eval", "iter": it + 1, "val_loss": round(head, 4),
                  "val_by_domain": {k: round(v, 4) for k, v in vls.items()}})

        if (it + 1) % cfg.train.sample_interval == 0:
            print("[sample] " + sample_text(model, tok, cfg, device).replace("\n", " ⏎ "))

        if (it + 1) % cfg.train.ckpt_interval == 0 or it + 1 == cfg.train.max_iters:
            # 磁盘预算：保留 N 个定期档 + latest，每个约 1.9GB。不够就先喊，别等写爆。
            keep = int(getattr(cfg.train, "keep_ckpts", 0) or 0)
            free_gb = shutil.disk_usage(out_dir).free / 2**30
            need_gb = 1.9 * (keep + 1 if keep > 0 else 2)
            if free_gb < need_gb:
                print(f"[train] ⚠️ 磁盘仅剩 {free_gb:.1f}GB，低于本次需要的约 {need_gb:.1f}GB")

            state = {"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                     "config": cfg.to_dict(), "rng_torch": torch.get_rng_state(),
                     "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                     "tokenizer_path": str(tok_path), "tokenizer_sha": tok_sha}
            latest = out_dir / "latest.pt"
            tmp = out_dir / ".latest.pt.tmp"
            # 先写临时文件再原子替换。直接覆盖 latest.pt 时若断电/崩溃，留下的
            # 是半个文件——torch.load 会直接抛异常，几十小时的训练成果全部作废。
            try:
                torch.save(state, tmp)
                os.replace(tmp, latest)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

            path = out_dir / f"ckpt_{it+1:06d}.pt"
            try:
                path.unlink(missing_ok=True)
                os.link(latest, path)       # 硬链接：同一 inode，零拷贝零额外空间
            except OSError:
                shutil.copy2(latest, path)  # 跨盘或不支持硬链接时退化

            if keep > 0:
                for old in sorted(out_dir.glob("ckpt_*.pt"))[:-keep]:
                    old.unlink(missing_ok=True)

            print(f"[train] 已保存 {path.name}（保留最近 {keep or '全部'} 个）")

    jlog({"type": "done", "iters": cfg.train.max_iters})
    print(f"[train] 完成：{cfg.train.max_iters} iters，最终 ckpt 在 {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
