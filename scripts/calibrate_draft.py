# -*- coding: utf-8 -*-
"""星辰 · MTP 草稿校准

投机解码的接受率取决于"MTP 草稿与主模型选择的吻合率"，而 MTP 在预训练里
学的是"真实下一个 token"——两者不是一回事。本脚本冻结主模型，只训练 MTP
分支去模仿主模型的 argmax（ KD 的硬标签版），直接优化接受率。

用法：python scripts/calibrate_draft.py --ckpt checkpoints-sft-v2/latest.pt --iters 600
完成后把校准后的模型另存为 <ckpt目录>/latest_spec.pt（主模型参数未被改动）。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from xingchen import XingChen, get_batch, load_bin
from xingchen.config import XingChenConfig, _merge_dataclass


@torch.no_grad()
def agreement(model, data, device, n_batch=8, seq=512):
    hits, total = 0, 0
    for _ in range(n_batch):
        high = len(data) - seq - 1
        ix = torch.randint(high, (2,))
        x = torch.stack([torch.from_numpy(data[i:i + seq].astype(np.int64)) for i in ix]).to(device)
        logits, h = model(x)
        m_logits = model.mtp_forward(h[:, :-1], x[:, 1:])
        # 草稿第 t 位预测 x_{t+2}；主模型第 t+1 位也预测 x_{t+2}
        hits += (m_logits[:, :-1].argmax(-1) == logits[:, 1:-1].argmax(-1)).sum().item()
        total += m_logits[:, :-1].argmax(-1).numel()
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints-sft-v2" / "latest.pt"))
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data_bin", default=None,
                    help="校准用 token 流；默认用 SFT 流（与模型实际行为分布一致）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(2026)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ckpt["config"])
    model = XingChen(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    data_bin = args.data_bin or (Path(args.ckpt).parent.parent / "data/sft/tokens.bin")
    data = load_bin(data_bin)
    print(f"[calibrate] 校准分布：{data_bin}（{len(data):,} token）")

    # 冻结主模型，只训练 MTP 分支
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mtp.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.mtp.parameters(), lr=args.lr, weight_decay=0.0)

    hit0 = agreement(model, data, device)
    print(f"[calibrate] 校准前 草稿-主模型 argmax 吻合率: {hit0:.1%}")

    model.train()
    for it in range(args.iters):
        x, _ = get_batch(data, args.batch_size, cfg.train.seq_len, device)
        with torch.no_grad():
            logits_main, h = model(x)
        m_logits = model.mtp_forward(h[:, :-1], x[:, 1:])
        target = logits_main[:, 1:-1].argmax(-1)     # 主模型对 t+2 的选择
        loss = F.cross_entropy(m_logits[:, :-1].reshape(-1, m_logits.size(-1)).float(),
                               target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0 or it == 0:
            print(f"[calibrate] iter {it+1}/{args.iters} | 模仿 loss {loss.item():.4f}")

    hit1 = agreement(model, data, device)
    print(f"[calibrate] 校准后 草稿-主模型 argmax 吻合率: {hit1:.1%}（前 {hit0:.1%}）")

    out = Path(args.ckpt).parent / "latest_spec.pt"
    torch.save({**ckpt, "model": model.state_dict(), "iter": ckpt.get("iter", 0),
                "note": "MTP 草稿已校准"}, out)
    print(f"[calibrate] 已保存 {out}")


if __name__ == "__main__":
    main()
