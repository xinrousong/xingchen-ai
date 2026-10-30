# -*- coding: utf-8 -*-
"""GPU 上比 MoE 两种分派的真实差距（同步次数才是瓶颈，CPU 基准看不出）。"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xingchen.model as M  # noqa: E402
from xingchen.config import load_config  # noqa: E402
from xingchen.model import MoEFeedForward  # noqa: E402

cfg = load_config(ROOT / "configs" / "base-wiki.yaml")
mcfg = cfg.model
rep = []

def out(s=""):
    rep.append(str(s)); print(s)

if not torch.cuda.is_available():
    out("no cuda"); raise SystemExit

dev = "cuda"
out(f"GPU: {torch.cuda.get_device_name(0)}  已占用 {torch.cuda.memory_allocated()/2**30:.2f} GB")
moe = MoEFeedForward(mcfg).to(dev).eval()
x = torch.randn(8, 128, mcfg.dim, device=dev)
out(f"MoE: dim={mcfg.dim} E={mcfg.moe.n_routed} top_k={mcfg.moe.top_k} N={x.numel()//mcfg.dim}")

for mode in ("loop", "vectorized"):
    M._MOE_DISPATCH = mode
    with torch.no_grad():
        for _ in range(3):
            moe(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            moe(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    out(f"  {mode:11s}: {dt*1e3:7.2f} ms/次")

Path(ROOT / "logs" / "fastpath_gpu.txt").write_text("\n".join(rep), encoding="utf-8")
