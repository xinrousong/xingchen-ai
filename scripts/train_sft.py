# -*- coding: utf-8 -*-
"""星辰 · SFT 监督微调：在预训练检查点基础上，只对"回答部分"计算损失

用法：
  python scripts/train_sft.py --init checkpoints-wiki/latest.pt --max_iters 3000

与预训练的区别：
* 数据是 (tokens, mask) 双流：mask=0 的位置（提示词部分）不计损失
* 学习率小一个量级（1e-4），不再需要长 warmup
* MTP 分支保留在权重里但不再训练（V3 二阶段做法）
"""
from __future__ import annotations

import argparse
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

from xingchen import XingChen, generate
from xingchen.config import XingChenConfig, _merge_dataclass
from xingchen.tokenizer import ByteBPETokenizer

SAMPLE_QUESTIONS = ["你好，请介绍一下你自己。", "地球为什么有四季变化？", "用一句话解释什么是人工智能。"]


def get_batch(tokens: np.memmap, mask: np.memmap, batch_size: int, seq_len: int,
              device: torch.device):
    seq = min(seq_len, len(tokens) - 1)
    high = len(tokens) - seq - 1
    ix = torch.randint(high, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + seq].astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + seq].astype(np.float32)) for i in ix])
    return x.to(device), y.to(device), m.to(device)


def get_lm_batch(tokens: np.memmap, batch_size: int, seq_len: int, device: torch.device):
    """replay 用：纯语言建模 batch，全部位置都算损失。"""
    seq = min(seq_len, len(tokens) - 1)
    high = len(tokens) - seq - 1
    ix = torch.randint(high, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i:i + seq].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + seq].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def masked_ce(logits, y, m):
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                         y.reshape(-1), reduction="none")
    return (ce * m.reshape(-1)).sum() / m.sum().clamp_min(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="预训练检查点（如 checkpoints-wiki/latest.pt）")
    ap.add_argument("--data_dir", default=str(ROOT / "data/sft"))
    ap.add_argument("--out_dir", default=str(ROOT / "checkpoints-sft"))
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--replay_bin", default=str(ROOT / "data/wiki/train.bin"),
                    help="预训练语料（replay / 混合训练用）")
    ap.add_argument("--replay_prob", type=float, default=0.2)
    ap.add_argument("--core_bin", default=str(ROOT / "data/wiki/core_topup.bin"),
                    help="核心记忆语料；和 SFT、预训练一起混着练")
    ap.add_argument("--p_core", type=float, default=0.0,
                    help="每步走核心记忆的概率（>0 时启用三路混合训练）")
    ap.add_argument("--p_pretrain", type=float, default=None,
                    help="每步走预训练语料的概率（默认沿用 --replay_prob）")
    ap.add_argument("--val_bin", default=str(ROOT / "data/wiki/val.bin"))
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ckpt["config"])
    cfg.model.mtp.loss_weight = 0.0        # 二阶段不再训练 MTP

    data_dir = Path(args.data_dir)
    tokens = np.memmap(data_dir / "tokens.bin", dtype=np.uint32, mode="r")
    mask = np.memmap(data_dir / "mask.bin", dtype=np.uint8, mode="r")
    tok = ByteBPETokenizer.load(data_dir / "tokenizer.json")
    print(f"[sft] 数据 {len(tokens):,} token，回答部分占比 {mask.mean()*100:.1f}%")

    # ---- 三路混合训练：SFT 问答 / 核心记忆 / 预训练语料
    # 分开两阶段训练一定会互相破坏（练记忆就把语言练坏，练语言就把记忆冲掉），
    # 所以干脆每一步都从三个分布里抽一个。
    p_pre = args.replay_prob if args.p_pretrain is None else args.p_pretrain

    def _mm(path: str, label: str):
        p = Path(path)
        if not p.exists():
            print(f"[sft] 警告：找不到{label}语料 {p}")
            return None
        return np.memmap(p, dtype=np.uint32, mode="r")

    mix = []                                   # [(name, memmap, prob)]
    if args.p_core > 0:
        core_mm = _mm(args.core_bin, "核心记忆")
        if core_mm is not None:
            mix.append(("core", core_mm, args.p_core))
    if p_pre > 0:
        pre_mm = _mm(args.replay_bin, "预训练")
        if pre_mm is not None:
            mix.append(("pretrain", pre_mm, p_pre))
    p_sft = max(0.0, 1.0 - sum(p for _, _, p in mix))
    print("[sft] 混合比例：SFT {:.0%}".format(p_sft)
          + "".join(f" / {n} {p:.0%}" for n, _, p in mix))
    counts = {"sft": 0, **{n: 0 for n, _, _ in mix}}
    mix_map = {n: mm for n, mm, _ in mix}
    for n, mm, _ in mix:
        print(f"[sft]   {n}: {len(mm):,} token")

    # ---- 验证集：固定切片，用来监控「有没有把预训练能力练坏」
    val = None
    vp = Path(args.val_bin)
    if vp.exists():
        val = np.memmap(vp, dtype=np.uint32, mode="r")

    def eval_val(n: int = 4) -> float | None:
        if val is None:
            return None
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for k in range(n):
                i = 100_000 + k * T
                x = torch.tensor(np.asarray(val[i:i + T]).astype(np.int64), device=device)[None]
                y = torch.tensor(np.asarray(val[i + 1:i + 1 + T]).astype(np.int64), device=device)[None]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, _ = model(x)
                tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)).item()
        model.train()
        return tot / n

    model = XingChen(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[sft] 从 {args.init}（iter {ckpt['iter']+1}）加载预训练权重")

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    def jlog(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    open(log_path, "w", encoding="utf-8").close()
    jlog({"type": "meta", "name": cfg.name + "-sft", "params_m": 0,
          "vocab": cfg.model.vocab_size, "seq": args.seq_len, "batch": args.batch_size,
          "max_iters": args.max_iters, "device": str(device)})

    use_amp = device.type == "cuda"
    B, T = args.batch_size, args.seq_len

    def sample_answers() -> None:
        model.eval()
        for q in SAMPLE_QUESTIONS:
            ids = tok.encode(f"问：{q}\n答：", add_bos=True)
            out = generate(model, ids, max_new_tokens=120, temperature=0.7,
                           top_k=50, top_p=0.95, repetition_penalty=1.1,
                           no_repeat_ngram_size=0, eos_id=tok.eos_id)
            print(f"[sft-sample] Q: {q}\n[sft-sample] A: {tok.decode(out[len(ids):])}")
        model.train()

    model.train()
    t0 = time.time()
    tokens_seen = 0
    loss_acc = 0.0
    for it in range(args.max_iters):
        lr = args.lr * min(1.0, (it + 1) / 100)   # 极短 warmup 后恒定
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        r = random.random()
        chosen, acc = "sft", 0.0
        for name, _, p in mix:
            acc += p
            if r < acc:
                chosen = name
                break
        counts[chosen] += 1
        if chosen == "sft":
            x, y, m = get_batch(tokens, mask, B, T, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x)
                loss = masked_ce(logits, y, m)
        else:
            x, y = get_lm_batch(mix_map[chosen], B, T, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if model.moe_layers:
            model.update_expert_bias(cfg.model.moe.balance_gamma)
        loss_acc += loss.item()
        tokens_seen += x.numel()

        if (it + 1) % 10 == 0 or it == 0:
            dt = time.time() - t0
            mix_str = "/".join(f"{n}{counts[n]}" for n in counts)
            print(f"[sft] iter {it:6d} | loss {loss_acc/ (10 if it else 1):.4f} | lr {lr:.2e} "
                  f"| {tokens_seen/max(dt,1e-6)/1000:.1f}k tok/s | 混合 {mix_str}")
            jlog({"type": "train", "iter": it + 1, "loss": round(loss_acc / (10 if it else 1), 4),
                  "mtp": None, "lr": lr, "tok_s": round(tokens_seen / max(dt, 1e-6)),
                  "load_dev": None,
                  "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None})
            t0 = time.time()
            tokens_seen = 0
            loss_acc = 0.0

        if (it + 1) % args.eval_every == 0:
            vl = eval_val()
            if vl is not None:
                print(f"[sft] iter {it:6d} | 验证 loss {vl:.4f}")
                jlog({"type": "eval", "iter": it + 1, "val_loss": round(vl, 4)})

        if (it + 1) % 1000 == 0:
            sample_answers()

        if (it + 1) % 1500 == 0 or it + 1 == args.max_iters:
            path = out_dir / f"ckpt_{it+1:06d}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": cfg.to_dict(),
                        "rng_torch": torch.get_rng_state(),
                        "rng_numpy": np.random.get_state(),
                        "rng_python": random.getstate(),
                        "tokenizer_path": str(ROOT / cfg.data.tokenizer_path)}, path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": cfg.to_dict(),
                        "rng_torch": torch.get_rng_state(),
                        "rng_numpy": np.random.get_state(),
                        "rng_python": random.getstate(),
                        "tokenizer_path": str(ROOT / cfg.data.tokenizer_path)},
                       out_dir / "latest.pt")
            print(f"[sft] 已保存 {path.name}")

    jlog({"type": "done", "iters": args.max_iters})
    sample_answers()
    print(f"[sft] 完成，检查点在 {out_dir/'latest.pt'}")


if __name__ == "__main__":
    main()
