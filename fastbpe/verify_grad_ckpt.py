# -*- coding: utf-8 -*-
"""M3.2 验证：梯度检查点开关前后 (a) loss/梯度是否等价 (b) 显存与速度差多少。

只读，不改任何检查点。结果写 logs/grad_ckpt_check.txt。
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xingchen.config import load_config  # noqa: E402
from xingchen.model import XingChen  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rep = []
def out(s=""):
    rep.append(str(s)); print(s, flush=True)

cfg = load_config(ROOT / "configs" / "base-wiki.yaml")
ckpt = torch.load(ROOT / "checkpoints-base-sft" / "latest.pt", map_location="cpu", weights_only=False)
model = XingChen(cfg.model).to("cuda")
model.load_state_dict(ckpt["model"])
V = cfg.model.vocab_size
out(f"模型 {sum(p.numel() for p in model.parameters())/1e6:.1f}M，检查点 iter={ckpt.get('iter')}")


def trial(seq, batch, use_ckpt, steps=2, seed=0, lr=0.0):
    torch.manual_seed(seed)
    model.grad_checkpoint = use_ckpt
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, V, (batch, seq), device="cuda")
    y = torch.randint(0, V, (batch, seq), device="cuda")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(loss.item())
    gn = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for p in model.parameters():
        p.grad = None
    return losses, gn, peak, dt


out("\n=== 等价性（seq 512, batch 2）===")
la, ga, pa, ta = trial(512, 2, False)
lb, gb, pb, tb = trial(512, 2, True)
out(f"  关：loss {[f'{x:.4f}' for x in la]}  梯度范数 {ga:.3f}  峰值 {pa:.2f} GB  {ta*1000:.0f} ms/步")
out(f"  开：loss {[f'{x:.4f}' for x in lb]}  梯度范数 {gb:.3f}  峰值 {pb:.2f} GB  {tb*1000:.0f} ms/步")
d_loss = max(abs(a - b) for a, b in zip(la, lb))
d_grad = abs(ga - gb) / max(ga, 1e-9)
out(f"  loss 最大差 {d_loss:.2e}   梯度相对差 {d_grad:.2e}   "
    f"{'✅ 等价' if d_loss < 5e-3 and d_grad < 5e-3 else '❌ 不一致'}")

out("\n=== 显存/速度（batch 2）===")
for seq, use_ckpt, label in ((512, False, "512 不检查点"), (512, True, "512 检查点"),
                             (1024, True, "1024 检查点"), (1024, False, "1024 不检查点")):
    try:
        _, _, peak, dt = trial(seq, 2, use_ckpt, steps=1)
        out(f"  {label:14s}: 峰值 {peak:.2f} GB   {dt*1000:5.0f} ms/步")
    except RuntimeError as exc:
        out(f"  {label:14s}: 失败（{str(exc)[:80]}）")

(ROOT / "logs" / "grad_ckpt_check.txt").write_text("\n".join(rep), encoding="utf-8")
out("\n写入 logs/grad_ckpt_check.txt")
