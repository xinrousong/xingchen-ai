# -*- coding: utf-8 -*-
"""上线前预检：用真实 base 检查点，逐项定位快速路径的数值差异（fp32 / bf16 分开看）。"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xingchen.model as M  # noqa: E402
from xingchen.config import load_config  # noqa: E402

cfg = load_config(ROOT / "configs" / "base-wiki.yaml")
ckpt = torch.load(ROOT / "checkpoints-base-wiki" / "latest.pt", map_location="cpu", weights_only=False)
model = M.XingChen(cfg.model).eval().to("cuda")
model.load_state_dict(ckpt["model"])
print(f"checkpoint iter={ckpt.get('iter')}")

torch.manual_seed(0)
V = cfg.model.vocab_size
x = torch.randint(0, V, (2, 129), device="cuda")
idx, tgt = x[:, :-1], x[:, 1:]          # 正常的下一个 token 预测

def run(moe, mla, bf16):
    M._MOE_DISPATCH, M._MLA_ATTN = moe, mla
    with torch.no_grad():
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(idx)
        else:
            logits, _ = model(idx)
    logits = logits.float()
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    return logits, loss.item()

for bf16 in (False, True):
    tag = "bf16" if bf16 else "fp32"
    base, lb = run("loop", "naive", bf16)
    print(f"\n[{tag}] 基准 loss = {lb:.4f}  (logits std = {base.std().item():.3f}, max|.| = {base.abs().max().item():.2f})")
    for moe, mla in (("vectorized", "naive"), ("loop", "sdpa"), ("vectorized", "sdpa")):
        out, lo = run(moe, mla, bf16)
        d = (out - base).abs().max().item()
        print(f"  {moe:10s}+{mla:5s}: loss {lo:.4f} (Δ{abs(lo-lb):.5f})  logits maxΔ {d:.3e}")
print("\nPREFLIGHT OK")
