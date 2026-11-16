# -*- coding: utf-8 -*-
"""分域困惑度评测（补上"只有一张考卷"的短板）

现有评测只有 `core_check.py`（31 题记忆考卷）和人工抽查，缺的是**连续指标**：
模型在不同语域上的 loss 到底涨还是跌，之前是看不出来的。这个脚本补这一块。

特点：
  * 支持多个验证流（每个语域一个 .bin），分别报 ppl；
  * 窗口起点固定（等距取），同一份 bin 在任何机器上结果一致，可跨机器对比；
  * 顺带报"按位置切段"的 ppl，用来判断长程能力是否塌（前段/中段/后段）；
  * bf16 autocast 与训练一致；CPU 也能跑（用于无卡时冒烟）。

用法：
    python scripts/eval_ppl.py --ckpt checkpoints-base-wiki/latest.pt \
        --bin zh_wiki=data/wiki/val.bin --iters 30

    # 多语域对比（1B 语料会有多个 val）
    python scripts/eval_ppl.py --ckpt checkpoints-base-1b/latest.pt \
        --bin zh_wiki=data/val/zh_wiki.bin --bin zh_books=data/val/zh_books.bin \
        --bin en_web=data/val/en_web.bin --bin code=data/val/code.bin
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xingchen.config import XingChenConfig, _merge_dataclass  # noqa: E402
from xingchen.model import XingChen                            # noqa: E402


def windows(n_tokens: int, seq: int, count: int) -> list[int]:
    """等距取窗口起点：结果是确定的，可跨机器/跨版本比较。"""
    high = n_tokens - seq - 1
    if high <= 0:
        return [0]
    step = max(1, high // count)
    return [min(step * k, high) for k in range(count)]


@torch.no_grad()
def eval_bin(model, path: Path, seq: int, iters: int, device, amp: bool) -> dict:
    data = np.memmap(path, dtype=np.uint32, mode="r")
    seq = min(seq, len(data) - 2)
    offsets = windows(len(data), seq, iters)

    losses = []
    for off in offsets:
        x = torch.from_numpy(np.asarray(data[off:off + seq], dtype=np.int64))[None].to(device)
        y = torch.from_numpy(np.asarray(data[off + 1:off + 1 + seq], dtype=np.int64))[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                      y.reshape(-1)).item())

    # 按窗口位置切三段，看长程是否塌（seq 越大越有意义）
    k = max(1, len(losses) // 3)
    head, tail = losses[:k], losses[-k:]
    mean = sum(losses) / len(losses)
    return {
        "bin": path.name,
        "tokens": len(data),
        "windows": len(losses),
        "loss": round(mean, 4),
        "ppl": round(math.exp(min(mean, 20)), 2),
        "loss_first_third": round(sum(head) / len(head), 4),
        "loss_last_third": round(sum(tail) / len(tail), 4),
        "loss_min": round(min(losses), 4),
        "loss_max": round(max(losses), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", action="append", required=True,
                    help="name=path，可重复；name 用于报告里区分语域")
    ap.add_argument("--seq", type=int, default=0, help="0=用权重里的 seq_len")
    ap.add_argument("--iters", type=int, default=30, help="每个 bin 取多少个窗口")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None, help="结果写入 JSON")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu"))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ck["config"])
    model = XingChen(cfg.model).to(device).eval()
    model.load_state_dict(ck["model"])
    seq = args.seq or cfg.train.seq_len
    amp = device.type == "cuda" and cfg.train.amp == "bf16"

    print(f"权重 {Path(args.ckpt).name} | iter {ck.get('iter')} | {cfg.name}")
    print(f"设备 {device} | seq {seq} | 每 bin {args.iters} 个窗口 | amp={'bf16' if amp else 'off'}")
    print(f"\n{'语域':<16} {'token':>12} {'loss':>8} {'ppl':>9} {'前段':>8} {'后段':>8} {'最差窗':>9}")
    rows, missing = [], []
    for spec in args.bin:
        if "=" not in spec:
            print(f"[WARN] 忽略格式错误的 --bin：{spec}（应写成 name=path）")
            continue
        name, p = spec.split("=", 1)
        path = Path(p)
        if not path.exists():
            missing.append(str(path))
            continue
        r = eval_bin(model, path, seq, args.iters, device, amp)
        r["name"] = name
        rows.append(r)
        print(f"{name:<16} {r['tokens']:>12,} {r['loss']:>8.4f} {r['ppl']:>9.2f} "
              f"{r['loss_first_third']:>8.4f} {r['loss_last_third']:>8.4f} {r['loss_max']:>9.4f}")

    if missing:
        print(f"\n[WARN] 以下 bin 不存在，已跳过：{missing}")
    if not rows:
        print("[FATAL] 没有任何可评测的 bin")
        return 2

    # 加权平均 ppl（按 token 数）
    tot = sum(r["tokens"] for r in rows)
    wloss = sum(r["loss"] * r["tokens"] for r in rows) / tot
    print(f"\n加权平均：loss {wloss:.4f}　ppl {math.exp(min(wloss, 20)):.2f}")

    worst = max(rows, key=lambda r: r["loss"])
    print(f"最差语域：{worst['name']}（ppl {worst['ppl']}）—— 下一批语料优先补这一域")

    print("\nPPL_JSON=" + json.dumps(rows, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
