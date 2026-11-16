# -*- coding: utf-8 -*-
"""micro-batch / 序列长度 / SDPA 后端 的扫描台（换卡或权限恢复后先跑这个）

目的：在改任何超参之前，先知道两件事——
  1. 注意力实际落在哪个 SDPA 后端（math 与 mem_efficient 能差 3~5 倍）；
  2. batch × seq 组合里，吞吐最高又不 OOM 的那一档在哪。

用法：
    python scripts/bench_microbatch.py --config configs/base-wiki-1b.yaml
    python scripts/bench_microbatch.py --config configs/base-wiki-1b.yaml --grad-accum 1

注意：
  * 只做 fwd+bwd 的干跑，不读语料、不存盘，不碰任何 checkpoint；
  * OOM 会被捕获并跳过该组合，不会中断整个扫描；
  * 结果最后会打印一行 BENCH_JSON，方便贴进报告。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xingchen.config import load_config  # noqa: E402
from xingchen.model import XingChen      # noqa: E402


def sdpa_backends() -> list[str]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return []
    names = []
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        if hasattr(SDPBackend, name):
            names.append(name)
    return names


def probe_backend(cfg, device) -> None:
    """逐个后端强制跑一次前向，看哪个真的能用。"""
    if not hasattr(torch.nn.attention, "sdpa_kernel"):
        print("[warn] 该 torch 没有 sdpa_kernel，跳过后端探测")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    model = XingChen(cfg.model).to(device).eval()
    ids = torch.randint(0, cfg.model.vocab_size, (1, 64), device=device)
    print("\n=== SDPA 后端可用性 ===")
    for name in sdpa_backends():
        backend = getattr(SDPBackend, name)
        try:
            with sdpa_kernel(backend), torch.no_grad():
                model(ids)
            print(f"  {name:20s} ✅ 可用")
        except Exception as e:
            print(f"  {name:20s} ❌ {type(e).__name__}: {str(e)[:60]}")
    del model
    torch.cuda.empty_cache()


def bench_one(cfg, device, batch: int, seq: int, steps: int = 3) -> dict | None:
    model = XingChen(cfg.model).to(device).train()
    model.grad_checkpoint = bool(cfg.train.grad_checkpoint)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.model.vocab_size, (batch, seq), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (batch, seq), device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(steps):                       # 第一步含编译/warmup，取后两步
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok = batch * seq * steps / dt
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        return {"batch": batch, "seq": seq, "tok_s": round(tok), "peak_gb": round(peak, 2)}
    except torch.cuda.OutOfMemoryError:
        return {"batch": batch, "seq": seq, "oom": True}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"batch": batch, "seq": seq, "oom": True}
        raise
    finally:
        del model, opt
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base-wiki-1b.yaml"))
    ap.add_argument("--batches", default="1,2,3,4")
    ap.add_argument("--seqs", default="512")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] 当前拿不到 CUDA（驱动权限或显存被占用），先解决再跑")
        return 2

    cfg = load_config(args.config)
    device = torch.device("cuda")
    print(f"配置 {cfg.name} | 设备 {torch.cuda.get_device_name(0)} | "
          f"torch {torch.__version__}")
    if not args.skip_probe:
        probe_backend(cfg, device)

    batches = [int(x) for x in args.batches.split(",")]
    seqs = [int(x) for x in args.seqs.split(",")]
    print("\n=== batch × seq 扫描（fwd+bwd 干跑）===")
    print(f"{'batch':>6} {'seq':>6} {'tok/s':>10} {'峰值显存':>10}")
    rows = []
    for seq in seqs:
        for b in batches:
            r = bench_one(cfg, device, b, seq, args.steps)
            if r is None:
                continue
            rows.append(r)
            if r.get("oom"):
                print(f"{b:>6} {seq:>6} {'OOM':>10} {'-':>10}")
            else:
                print(f"{b:>6} {seq:>6} {r['tok_s']:>10,} {r['peak_gb']:>9.2f}GB")
            torch.cuda.empty_cache()

    ok = [r for r in rows if not r.get("oom")]
    if ok:
        best = max(ok, key=lambda r: r["tok_s"])
        print(f"\n最佳：batch={best['batch']} seq={best['seq']} → {best['tok_s']:,} tok/s"
              f"（{best['peak_gb']:.2f} GB）")
    print("\nBENCH_JSON=" + json.dumps({"config": cfg.name, "runs": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
