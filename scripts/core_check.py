# -*- coding: utf-8 -*-
"""星辰 · 底层认知自检

用 data/core/qa/ 里的固定题目考模型：keywords 里任一词出现在回答中即判 ✅。
没有 keywords 的题只展示回答供人工阅读。报告同时写入 logs/core_check.log。

用法：
  python scripts/core_check.py                          # 默认考 checkpoints-sft-v2
  python scripts/core_check.py --ckpt checkpoints-sft-v2/latest.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from xingchen import XingChen, generate
from xingchen.config import XingChenConfig, _merge_dataclass
from xingchen.tokenizer import ByteBPETokenizer

LOG = ROOT / "logs" / "core_check_history.log"


def log(line: str) -> None:
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints-sft-v2" / "latest.pt"))
    ap.add_argument("--qa_dir", default=str(ROOT / "data/core/qa"))
    ap.add_argument("--max_new", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--rep", type=float, default=1.0,
                    help="重复惩罚；考试测的是记忆，默认 1.0（惩罚会把背熟的答案推偏）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _merge_dataclass(XingChenConfig, ckpt["config"])
    model = XingChen(cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer_path"])

    qa_dir = Path(args.qa_dir)
    questions: list[dict] = []
    for p in sorted(qa_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("q") and row.get("a"):
                questions.append(row)

    log(f"\n===== 认知自检 {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{args.ckpt}（iter {ckpt['iter']+1}）· {len(questions)} 题 =====")

    n_pass = n_fail = n_manual = 0
    for i, row in enumerate(questions, 1):
        ids = tok.encode(f"问：{row['q']}\n答：", add_bos=True)
        out = generate(model, ids, max_new_tokens=args.max_new,
                       temperature=args.temperature, top_k=50, top_p=0.95,
                       repetition_penalty=args.rep, no_repeat_ngram_size=0,
                       eos_id=tok.eos_id)
        answer = tok.decode(out[len(ids):]).strip().replace("\n", " ")
        kws = row.get("keywords") or []
        if kws:
            hit = [k for k in kws if k in answer]
            ok = bool(hit)
            n_pass += ok
            n_fail += not ok
            mark = "✅" if ok else "❌"
            log(f"{mark} Q{i}：{row['q']}")
            log(f"   答：{answer[:120] or '（空）'}")
            if not ok:
                log(f"   （期待出现：{'/'.join(kws)}）")
        else:
            n_manual += 1
            log(f"👁 Q{i}：{row['q']}（人工阅读题）")
            log(f"   答：{answer[:120] or '（空）'}")

    log(f"----- 结果：关键词题 {n_pass}✅ / {n_fail}❌，人工题 {n_manual} 道 -----")
    if n_fail == 0 and n_pass > 0:
        log("结论：底层认知在（记忆完好）。")
    elif n_pass == 0:
        log("结论：底层认知未建立或已遗忘——需要重跑训练（auto_train.py）。")
    else:
        log("结论：部分认知在——正常现象，理解是随训练批次慢慢变深的。")


if __name__ == "__main__":
    main()
