# -*- coding: utf-8 -*-
"""快速路径（MoE 分组分派 / MLA SDPA）与朴素实现的一致性校验 + CPU 基准。

只读：不加载任何检查点，随机初始化小模型；结果写 logs/fastpath_check.txt。
"""
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xingchen.model as M  # noqa: E402
from xingchen.config import load_config  # noqa: E402
from xingchen.model import MLAAttention, MoEFeedForward  # noqa: E402

rep = []
def out(s=""):
    rep.append(str(s))

torch.manual_seed(0)
cfg = load_config(ROOT / "configs" / "tiny.yaml")
mcfg = cfg.model
out(f"配置：dim={mcfg.dim} layers={mcfg.n_layers} experts={mcfg.moe.n_routed} "
    f"top_k={mcfg.moe.top_k} expert_dim={mcfg.moe.expert_dim}")

# ---------------- MoE：loop vs grouped ----------------
moe_old = MoEFeedForward(mcfg).eval()
moe_new = deepcopy(moe_old)
x = torch.randn(64, 16, mcfg.dim)          # 1024 token
moe_old.train(); moe_new.train()

M._MOE_DISPATCH = "loop"
y_old = moe_old(x)
M._MOE_DISPATCH = "vectorized"
y_new = moe_new(x)
diff = (y_old - y_new).abs().max().item()
out("")
out(f"[MoE] 前向最大差 = {diff:.3e}   {'✅' if diff < 1e-5 else '❌'}")

# 反传：梯度也要一致
moe_old.zero_grad(); y_old.sum().backward()
g_old = [p.grad.clone() for p in moe_old.parameters() if p.grad is not None]
moe_new.zero_grad(); y_new.sum().backward()
g_new = [p.grad.clone() for p in moe_new.parameters() if p.grad is not None]
gdiff = max((a - b).abs().max().item() for a, b in zip(g_old, g_new))
out(f"[MoE] 梯度最大差 = {gdiff:.3e}   {'✅' if gdiff < 1e-5 else '❌'}")
out(f"[MoE] 专家计数 loop={moe_old.expert_counts.tolist()} "
    f"vectorized={moe_new.expert_counts.tolist()}")

# ---------------- MLA：naive vs sdpa ----------------
att_old = MLAAttention(mcfg).eval()
att_new = deepcopy(att_old)
xa = torch.randn(2, 32, mcfg.dim)
cos = torch.randn(32, mcfg.mla.qk_rope_head_dim // 2).abs() + 0.5
sin = torch.randn(32, mcfg.mla.qk_rope_head_dim // 2) * 0.3

M._MLA_ATTN = "naive"
o_old = att_old(xa, cos, sin)
M._MLA_ATTN = "sdpa"
o_new = att_new(xa, cos, sin)
d_attn = (o_old - o_new).abs().max().item()
scale = o_old.abs().max().item()
out("")
out(f"[MLA] 前向最大差 = {d_attn:.3e}（量级 {scale:.3f}）   {'✅' if d_attn < 1e-4 else '❌'}")

att_old.zero_grad(); o_old.sum().backward()
ga = [p.grad.clone() for p in att_old.parameters() if p.grad is not None]
att_new.zero_grad(); o_new.sum().backward()
gb = [p.grad.clone() for p in att_new.parameters() if p.grad is not None]
d_ga = max((a - b).abs().max().item() for a, b in zip(ga, gb))
out(f"[MLA] 梯度最大差 = {d_ga:.3e}   {'✅' if d_ga < 1e-4 else '❌'}")

# ---------------- 小模型端到端 ----------------
m1 = M.XingChen(mcfg).eval()
m2 = deepcopy(m1)
idx = torch.randint(0, mcfg.vocab_size, (2, 32))
M._MOE_DISPATCH, M._MLA_ATTN = "loop", "naive"
l1, _ = m1(idx)
M._MOE_DISPATCH, M._MLA_ATTN = "vectorized", "sdpa"
l2, _ = m2(idx)
d_full = (l1 - l2).abs().max().item()
out("")
out(f"[整模型] logits 最大差 = {d_full:.3e}   {'✅' if d_full < 1e-3 else '❌'}")

# ---------------- CPU 基准（只为看同步/调度开销的比例） ----------------
xb = torch.randn(8, 128, mcfg.dim)
for mode in ("loop", "vectorized"):
    M._MOE_DISPATCH = mode
    moe = MoEFeedForward(mcfg).eval()
    with torch.no_grad():
        moe(xb)
        t0 = time.perf_counter()
        for _ in range(3):
            moe(xb)
        dt = (time.perf_counter() - t0) / 3
    out("")
    out(f"[基准] MoE 1024 token（CPU）{mode:11s}: {dt*1000:7.1f} ms")

Path(ROOT / "logs" / "fastpath_check.txt").write_text("\n".join(rep), encoding="utf-8")
print("written logs/fastpath_check.txt")
