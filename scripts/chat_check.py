# -*- coding: utf-8 -*-
"""聊天抽查：用真实会话来验收（比固定考卷更能看出「像不像个人」）。

用法：
    python scripts/chat_check.py --ckpt checkpoints-sft-mix/latest.pt --sft
    python scripts/chat_check.py --ckpt ... --sft --out logs/chat_check_mix.log

默认问的就是用户实测过的那几个问题，方便前后对比。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xingchen.config import load_config  # noqa: E402
from xingchen.generation import generate  # noqa: E402
from xingchen.model import XingChen  # noqa: E402
from xingchen.tokenizer import ByteBPETokenizer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS = [
    "我是谁？",
    "讲讲修月亮的人",
    "你不会什么",
    "今天天气怎么样",
    "今天是什么时候",
    "你的创作者",
    "你能预知明天的彩票号码吗？",
    "请证明黎曼猜想。",
    "你能做什么？",
    "晚安",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base-wiki.yaml"))
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = XingChen(cfg.model).eval().to("cuda")
    model.load_state_dict(ckpt["model"])
    tok = ByteBPETokenizer.load(ckpt.get("tokenizer_path") or str(ROOT / "data/wiki/tokenizer.json"))

    lines = [f"===== chat check {args.ckpt} (temp {args.temperature}) ====="]
    for q in QUESTIONS:
        prompt = f"问：{q}\n答：" if args.sft else q
        ids = tok.encode(prompt, add_bos=True)
        out = generate(model, ids, max_new_tokens=args.max_new, temperature=args.temperature,
                       top_k=50, top_p=0.95, repetition_penalty=1.1,
                       no_repeat_ngram_size=0, eos_id=tok.eos_id)
        ans = tok.decode(out[len(ids):]).strip().replace("\n", " ")
        lines.append(f"你> {q}")
        lines.append(f"星辰> {ans}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
