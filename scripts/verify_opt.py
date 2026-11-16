# -*- coding: utf-8 -*-
"""优化改动的等价性验证（CPU 可跑，不需要 GPU）

改了三处内部实现，必须证明「数值没变」才敢拿去训 1B：

  1. Expert 的 gate/up 合并成一次 GEMM      → 与拆分权重手算逐位一致
  2. ce_loss 分块计算                        → 与一次性 cross_entropy 差 < 1e-6
  3. get_batch 改一次花式索引                → 同种子下与旧实现取到完全相同的 x/y

额外检查：旧 checkpoint 能否 strict 加载（靠 Expert._load_from_state_dict 的拼接）、
参数量是否仍是 166.37M、前向能否跑通。

用法：
    python scripts/verify_opt.py                      # 用 checkpoints-base-wiki/latest.pt
    python scripts/verify_opt.py --ckpt checkpoints-sft-mix2/latest.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xingchen.config import XingChenConfig, _merge_dataclass  # noqa: E402
from xingchen.data import get_batch, load_bin                 # noqa: E402
from xingchen.model import XingChen, Expert                   # noqa: E402
from scripts.train import ce_loss                             # noqa: E402

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'✅' if ok else '❌'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def old_get_batch(data, batch_size, seq_len):
    """改动前的实现，用于对照。"""
    seq = min(seq_len, len(data) - 1)
    high = len(data) - seq
    ix = torch.randint(high, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i:i + seq], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + 1 + seq], dtype=np.int64)) for i in ix])
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints-base-wiki" / "latest.pt"))
    ap.add_argument("--val_bin", default=str(ROOT / "data" / "wiki" / "val.bin"))
    args = ap.parse_args()

    ck_path = Path(args.ckpt)
    if not ck_path.exists():
        print(f"[FATAL] 找不到权重 {ck_path}")
        return 2
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ck["config"])
    print(f"权重 {ck_path.name}  iter={ck.get('iter')}  name={cfg.name}\n")

    # ---------- 1. 旧 checkpoint 能否 strict 加载
    model = XingChen(cfg.model).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    check("旧 checkpoint strict 加载（gate/up 自动拼接）",
          not missing and not unexpected,
          f"missing={len(missing)} unexpected={len(unexpected)}")
    n_params = sum(p.numel() for p in model.parameters())
    check("参数量未变（166.37M）", abs(n_params / 1e6 - 166.37) < 0.05, f"{n_params/1e6:.2f}M")

    # ---------- 2. gate/up 合并的数值等价
    torch.manual_seed(0)
    dim, hidden = 64, 96
    exp = Expert(dim, hidden).eval()
    x = torch.randn(7, dim)
    with torch.no_grad():
        fused = exp(x)
        # 用合并后的权重拆回两半，手算参考值
        gu = exp.gate_up_proj.weight
        g_w, u_w = gu[:hidden], gu[hidden:]
        ref = F.linear(F.silu(F.linear(x, g_w)) * F.linear(x, u_w), exp.down_proj.weight)
    diff = (fused - ref).abs().max().item()
    check("gate/up 合并 ≡ 两个独立 Linear", diff < 1e-6, f"max|Δ|={diff:.2e}")

    # ---------- 3. ce_loss 分块等价
    torch.manual_seed(0)
    logits = torch.randn(64, 512, dtype=torch.bfloat16)
    target = torch.randint(0, 512, (64,))
    full = F.cross_entropy(logits.reshape(-1, 512).float(), target.reshape(-1))
    for chunk in (16, 256, 5000):
        got = ce_loss(logits, target, chunk=chunk)
        d = abs(got.item() - full.item())
        check(f"ce_loss(chunk={chunk}) ≡ 一次性 CE", d < 1e-6, f"Δ={d:.2e}")

    # ---------- 4. get_batch 同种子取到同一批
    val = load_bin(args.val_bin)
    torch.manual_seed(2026)
    x_new, y_new = get_batch(val, 4, 128, torch.device("cpu"))
    torch.manual_seed(2026)
    x_old, y_old = old_get_batch(val, 4, 128)
    check("get_batch 与旧实现同种子同数据",
          torch.equal(x_new, x_old) and torch.equal(y_new, y_old),
          f"形状 {tuple(x_new.shape)}")
    check("y 是 x 右移一位", torch.equal(x_new[:, 1:], y_new[:, :-1]))

    # ---------- 5. 前向能跑通（CPU、短序列）
    with torch.no_grad():
        ids = torch.randint(0, cfg.model.vocab_size, (1, 16))
        logits_out, h = model(ids)
    check("前向输出形状", tuple(logits_out.shape) == (1, 16, cfg.model.vocab_size),
          f"{tuple(logits_out.shape)}")
    check("前向无 NaN/Inf", torch.isfinite(logits_out).all().item())

    print()
    if FAIL:
        print(f"[FAIL] {len(FAIL)} 项未通过：{FAIL}")
        return 1
    print("[PASS] 三处改动均数值等价，旧权重可直接加载 —— 可以用于新语料训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
