# -*- coding: utf-8 -*-
"""星辰 · M0.5 固定考卷（评测基准）

用法：
    python scripts/eval_suite.py --ckpt checkpoints-base-sft/latest.pt
    python scripts/eval_suite.py --ckpt ... --sft --max_new 80

判分规则（刻意保持朴素、可复核）：
  * 每题有一个 keywords 列表，回答里**出现任意一个**即判 ✅（大小写不敏感）
  * 没有 keywords 的题（data/core/qa 里的手写题）只打印回答，人工读
  * 结果追加写入 logs/eval_history.log，一行一次，便于横向比不同检查点

考卷来源：
  * data/core/qa/*.jsonl        —— 她的记忆（复用 core_check 的题）
  * data/eval/qa_general.jsonl  —— 常识 / 英文 / 代码 / 诚实边界
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xingchen.model as M  # noqa: E402
from xingchen.config import load_config  # noqa: E402
from xingchen.generation import generate  # noqa: E402
from xingchen.model import XingChen  # noqa: E402
from xingchen.tokenizer import ByteBPETokenizer  # noqa: E402

HISTORY = ROOT / "logs" / "eval_history.log"


def load_questions() -> list[dict]:
    rows: list[dict] = []
    for p in sorted((ROOT / "data" / "core" / "qa").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                rows.append({"id": d.get("id", ""), "cat": "记忆",
                             "q": d["q"], "keywords": d.get("keywords") or []})
    p = ROOT / "data" / "eval" / "qa_general.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base-wiki.yaml"))
    ap.add_argument("--sft", action="store_true", help="按 SFT 的问答格式提示")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = XingChen(cfg.model).eval().to(args.device)
    model.load_state_dict(ckpt["model"])
    tok = ByteBPETokenizer.load(ckpt.get("tokenizer_path") or str(ROOT / "data/wiki/tokenizer.json"))

    rows = load_questions()
    ok = 0
    scored = 0
    per_cat: dict[str, list[int]] = {}
    lines: list[str] = []
    t0 = time.time()

    for r in rows:
        prompt = f"问：{r['q']}\n答：" if args.sft else r["q"]
        ids = tok.encode(prompt, add_bos=True)
        out = generate(model, ids, max_new_tokens=args.max_new, temperature=args.temperature,
                       top_k=1 if args.temperature <= 0 else 50, top_p=0.95,
                       repetition_penalty=1.0 if args.temperature <= 0 else 1.1,
                       eos_id=tok.eos_id)
        ans = tok.decode(out[len(ids):]).strip()
        kws = [k.lower() for k in r.get("keywords") or []]
        if kws:
            scored += 1
            hit = any(k in ans.lower() for k in kws)
            ok += int(hit)
            per_cat.setdefault(r["cat"], [0, 0])
            per_cat[r["cat"]][0] += int(hit)
            per_cat[r["cat"]][1] += 1
            mark = "✅" if hit else "❌"
        else:
            mark = "·"
        lines.append(f"{mark} [{r['cat']}] {r['q']}\n   答：{ans[:160]}")

    head = (f"\n===== eval {time.strftime('%Y-%m-%d %H:%M')} =====\n"
            f"ckpt={Path(args.ckpt).name} iter={ckpt.get('iter')} sft={args.sft} "
            f"temp={args.temperature} 关键词题 {ok}/{scored} "
            f"({ok/max(1,scored)*100:.0f}%)  用时 {time.time()-t0:.0f}s")
    cat_line = "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per_cat.items()))
    body = "\n".join(lines)
    print(head)
    print("分项：" + cat_line)
    print(body)
    HISTORY.parent.mkdir(exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(head + "\n分项：" + cat_line + "\n" + body + "\n")


if __name__ == "__main__":
    main()
