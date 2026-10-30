# -*- coding: utf-8 -*-
"""星辰 · 对话 / 续写脚本

用法：
  python scripts/chat.py --ckpt checkpoints/latest.pt                # 交互模式
  python scripts/chat.py --ckpt checkpoints/latest.pt --prompt "星辰" # 单次续写

注意：这是预训练 base 模型（未做 SFT），行为是"续写"而非"指令跟随"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

import torch

from xingchen import XingChen, generate
from xingchen.config import XingChenConfig, _merge_dataclass
from xingchen.tokenizer import ByteBPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "latest.pt"))
    ap.add_argument("--prompt", default=None, help="给出则单次生成后退出")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.1, help="重复惩罚")
    ap.add_argument("--sft", action="store_true",
                    help="对话模板模式：把输入包成「问：…答：」，用于 SFT 后的模型")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ckpt["config"])
    model = XingChen(cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer_path"])
    print(f"[chat] 已加载 {args.ckpt}（iter {ckpt['iter']+1}，{cfg.name}），设备 {device}")
    print(f"[chat] 采样：temp={args.temperature} top_k={args.top_k} top_p={args.top_p} rep={args.rep}")

    def reply(prompt: str) -> str:
        text = f"问：{prompt}\n答：" if args.sft else prompt
        ids = tok.encode(text, add_bos=True)
        out = generate(model, ids, max_new_tokens=args.max_new,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.rep, eos_id=tok.eos_id)
        return tok.decode(out[len(ids):])

    if args.prompt is not None:
        print(reply(args.prompt))
        return

    print("[chat] 进入续写模式，输入文本回车生成，输入 exit 退出。")
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("exit", "quit", "q"):
            break
        if not line:
            continue
        print("星辰> " + reply(line))


if __name__ == "__main__":
    main()
