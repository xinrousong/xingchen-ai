# -*- coding: utf-8 -*-
"""等 M1.5 训练链（logs/m15_chain.log 出现 "chain DONE"）结束后，自动跑固定考卷。

跑两个模型，方便横向对比：
  1. base-SFT（checkpoints-base-sft/latest.pt）—— 有对话训练
  2. base 原版（checkpoints-base-wiki/latest.pt）—— 只做过预训练
结果追加到 logs/eval_history.log，各自的控制台输出在 logs/eval_<标签>.log。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN_LOG = ROOT / "logs" / "m15_chain.log"
POLL_SECONDS = 60

TARGETS = [
    ("base-sft", "checkpoints-base-sft/latest.pt"),
    ("base", "checkpoints-base-wiki/latest.pt"),
]


def chain_done() -> bool:
    if not CHAIN_LOG.exists():
        return False
    text = CHAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    return "chain DONE" in text or "FAILED" in text


def main() -> int:
    print(f"[watch] waiting for {CHAIN_LOG.name} ...", flush=True)
    while not chain_done():
        time.sleep(POLL_SECONDS)
    print("[watch] chain finished, starting eval", flush=True)

    py = sys.executable
    for label, ckpt in TARGETS:
        if not (ROOT / ckpt).exists():
            print(f"[watch] skip {label}: {ckpt} 不存在", flush=True)
            continue
        out = ROOT / "logs" / f"eval_{label}.log"
        with open(out, "ab") as fh:
            fh.write(f"\n=== eval {label} {time.strftime('%m-%d %H:%M:%S')} ===\n".encode())
            subprocess.run(
                [py, "-u", "scripts/eval_suite.py", "--ckpt", ckpt, "--sft",
                 "--temperature", "0", "--max_new", "60"],
                cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
            )
        print(f"[watch] {label} done -> {out.name}", flush=True)
    print("[watch] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
