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
import dataclasses
import json
import math
import random
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
    train_data = load_bin(ROOT / cfg.data.train_bin)
    val_data = load_bin(ROOT / cfg.data.val_bin)
    print(f"[train] 训练 token {len(train_data):,} / 验证 token {len(val_data):,}")

    replay_data = None
    if args.replay_bin:
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

    open(log_path, "w", encoding="utf-8").close()
    jlog({"type": "meta", "name": cfg.name, "params_m": round(stats["total"] / 1e6, 2),
          "vocab": cfg.model.vocab_size, "seq": T, "batch": B,
          "max_iters": cfg.train.max_iters, "device": str(device)})

    def eval_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(cfg.train.eval_iters):
                x, y = get_batch(val_data, B, T, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, _ = model(x)
                    losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                                  y.reshape(-1)).item())
        model.train()
        return sum(losses) / len(losses)

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
            src = train_data
            if replay_data is not None and random.random() < args.replay_prob:
                src = replay_data          # 掺一条预训练批次，保住通用语言能力
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        max_dev = model.update_expert_bias(cfg.model.moe.balance_gamma) if model.moe_layers else None

        if (it + 1) % cfg.train.log_interval == 0 or it == start_iter:
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-6)
            mtp_str = f" | mtp {loss_mtp_acc / cfg.train.grad_accum:.3f}" if loss_mtp_acc else ""
            bal_str = f" | 负载偏差 {max_dev:.3f}" if max_dev is not None else ""
            mem = (f" | 显存 {torch.cuda.max_memory_allocated()/2**30:.2f}GB" if device.type == "cuda" else "")
            print(f"[train] iter {it:6d} | loss {loss_main_acc/cfg.train.grad_accum:.4f}{mtp_str}"
                  f" | lr {lr:.2e} | {tps/1000:.1f}k tok/s{bal_str}{mem}")
            jlog({"type": "train", "iter": it + 1,
                  "loss": round(loss_main_acc / cfg.train.grad_accum, 4),
                  "mtp": round(loss_mtp_acc / cfg.train.grad_accum, 4) if loss_mtp_acc else None,
                  "lr": lr, "tok_s": round(tps),
                  "load_dev": round(max_dev, 4) if max_dev is not None else None,
                  "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None})
            t0 = time.time()
            tokens_seen = 0

        if (it + 1) % cfg.train.eval_interval == 0:
            vl = eval_val()
            print(f"[train] iter {it:6d} | 验证 loss {vl:.4f}")
            jlog({"type": "eval", "iter": it + 1, "val_loss": round(vl, 4)})

        if (it + 1) % cfg.train.sample_interval == 0:
            print("[sample] " + sample_text(model, tok, cfg, device).replace("\n", " ⏎ "))

        if (it + 1) % cfg.train.ckpt_interval == 0 or it + 1 == cfg.train.max_iters:
            path = out_dir / f"ckpt_{it+1:06d}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": cfg.to_dict(), "rng_torch": torch.get_rng_state(),
                        "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                        "tokenizer_path": str(tok_path)}, path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": cfg.to_dict(), "rng_torch": torch.get_rng_state(),
                        "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                        "tokenizer_path": str(tok_path)}, out_dir / "latest.pt")
            print(f"[train] 已保存 {path.name}")

    jlog({"type": "done", "iters": cfg.train.max_iters})
    print(f"[train] 完成：{cfg.train.max_iters} iters，最终 ckpt 在 {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
