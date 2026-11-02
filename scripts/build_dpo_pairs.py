# -*- coding: utf-8 -*-
"""M1.6 第一步：用当前模型自己采样，再用规则打分挑出「好/坏」回答，做成 DPO 偏好对。

为什么不用人工标注：这台机器上没有标注数据，而 166M 的模型最该先学的是
「回答要完整、不空洞、不复读」这类**格式与态度**，这些用规则就能判。

输出：data/dpo/pairs.jsonl，每行 {prompt, chosen, rejected, gap}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
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

CJK = re.compile(r"[\u4e00-\u9fff]")


def score(ans: str, prompt: str) -> float:
    """粗糙但可控的打分：长度、重复、空洞、是否复述问题。"""
    s = 0.0
    n = len(ans.strip())
    if n == 0:
        return -5.0
    if 12 <= n <= 220:
        s += 1.0
    elif n < 8:
        s -= 1.5
    # 4-gram 重复率
    if n >= 20:
        grams = [ans[i:i + 4] for i in range(n - 3)]
        ratio = 1.0 - len(set(grams)) / len(grams)
        if ratio < 0.15:
            s += 1.0
        elif ratio > 0.5:
            s -= 2.0
    # 中文占比太低 → 可能跑偏
    cjk = len(CJK.findall(ans))
    if cjk < 6:
        s -= 1.0
    # 结尾没断（被截断的迹象）
    if ans.strip()[-1] in "，、：,：的了下":
        s -= 0.5
    return s


def load_prompts(limit: int, seed: int) -> list[str]:
    src = Path(r"E:\xingchen\data\sft_raw\alpaca-zh.json")
    if not src.exists():
        src = ROOT / "data" / "sft_raw" / "alpaca-zh.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    instr = [r.get("instruction", "").strip() for r in rows if r.get("instruction")]
    instr = [x for x in instr if 4 <= len(x) <= 80]
    random.Random(seed).shuffle(instr)
    return instr[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints-base-sft" / "latest.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    ap.add_argument("--prompts", type=int, default=150)
    ap.add_argument("--samples", type=int, default=4, help="每个提示采样几条")
    ap.add_argument("--max_new", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="在已有 pairs.jsonl 上续跑：跳过已用过的提示，追加新对")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "base-wiki.yaml")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = XingChen(cfg.model).eval().to("cuda")
    model.load_state_dict(ckpt["model"])
    tok = ByteBPETokenizer.load(ckpt.get("tokenizer_path") or str(ROOT / "data/wiki/tokenizer.json"))

    prompts = load_prompts(args.prompts, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.parent / "progress.txt"

    used: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    used.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    pass
        progress_path.write_text("", encoding="utf-8")
    else:
        used = set()
        progress_path.write_text("", encoding="utf-8")
        out_path.write_text("", encoding="utf-8")

    t0 = time.time()
    made = 0
    seen = 0
    skipped = 0

    def report(msg: str) -> None:
        """同时写进度文件（供另一个窗口 tail -Wait 实时看）。"""
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")
        print(f"[pairs] {msg}", flush=True)

    if used:
        report(f"续跑：已有 {len(used)} 条提示做过，跳过它们继续")

    with open(out_path, "a" if used else "w", encoding="utf-8") as fh:
        for i, q in enumerate(prompts, 1):
            prefix = f"问：{q}\n答："
            if prefix in used:          # 续跑：这条提示已经有对了
                skipped += 1
                continue
            ids = tok.encode(prefix, add_bos=True)
            cands = []
            for k in range(args.samples):
                out = generate(model, ids, max_new_tokens=args.max_new,
                               temperature=0.9 if k else 0.7, top_k=50, top_p=0.95,
                               repetition_penalty=1.1, eos_id=tok.eos_id)
                ans = tok.decode(out[len(ids):]).strip()
                if ans:
                    cands.append((score(ans, q), ans))
            if len(cands) >= 2:
                seen += 1
                cands.sort(key=lambda x: x[0])
                worst, best = cands[0], cands[-1]
                if best[0] - worst[0] >= 1.0:      # 差距太小，不适合当偏好对
                    fh.write(json.dumps({"prompt": prefix, "chosen": best[1], "rejected": worst[1],
                                         "chosen_score": round(best[0], 2),
                                         "rejected_score": round(worst[0], 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()                      # 立刻落盘，进度可见、崩了也不丢
                    made += 1
            elapsed = time.time() - t0
            done_now = i - skipped
            eta = elapsed / max(1, done_now) * (len(prompts) - i) / 60
            if done_now % 5 == 0 or i == len(prompts):
                report(f"{i}/{len(prompts)} 提示 | 本次新增 {made} 对 | 跳过 {skipped} | "
                       f"已用 {elapsed/60:.0f} 分钟 | 剩余约 {eta:.0f} 分钟")

    report(f"完成：本次新增 {made} 对，跳过 {skipped} 条已做过的（{seen} 个提示有效）")


if __name__ == "__main__":
    main()
