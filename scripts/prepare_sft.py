# -*- coding: utf-8 -*-
"""星辰 · SFT 数据准备：开源指令数据 → 带损失掩码的 token 流

下载 shibing624/alpaca-zh（中文指令数据），每条样本编码为：
  问：{instruction}（{input}）\n答：{output}<eos>
其中"问：…答："之前的部分掩码为 0（不参与损失），回答部分掩码为 1（参与损失）
——这样模型学的是"如何回答"，而不是"如何提问"。

全部样本拼接成一条流存 tokens.bin，对应掩码存 mask.bin（uint8）。

用法：python scripts/prepare_sft.py --out_dir data/sft
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from xingchen.tokenizer import ByteBPETokenizer

# shibing624/alpaca-zh 的主数据文件（GPT-4 翻译版 alpaca，约 52k 条；走 hf-mirror）
DATA_URL = "https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json"


def download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"[sft] 已存在：{dst}")
        return dst
    print(f"[sft] 下载 {url}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "xingchen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"[sft] 完成 {dst.stat().st_size/1e6:.1f} MB，用时 {time.time()-t0:.0f}s")
    return dst


def make_example(instruction: str, inp: str, output: str) -> tuple[str, str]:
    """返回 (不参与损失的提示部分, 参与损失的回答部分)。"""
    q = instruction.strip()
    if inp.strip():
        q += f"\n补充信息：{inp.strip()}"
    a = output.strip()
    return f"问：{q}\n答：", f"{a}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data/sft"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/wiki/tokenizer.json"),
                    help="沿用预训练分词器，保证词表一致")
    ap.add_argument("--max_examples", type=int, default=0, help="0=全部")
    ap.add_argument("--core_qa_dir", default=str(ROOT / "data/core/qa"),
                    help="核心问答目录（关于宪法与故事的题目，随 SFT 反复训练）")
    ap.add_argument("--core_qa_repeat", type=int, default=120,
                    help="核心问答重复份数（理解强度；57M 模型需要百余次才能记住）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = download(DATA_URL, ROOT / "data/sft_raw/alpaca-zh.json")
    data = json.loads(raw.read_text(encoding="utf-8"))
    print(f"[sft] 指令样本 {len(data):,} 条")
    if args.max_examples:
        data = data[:args.max_examples]

    # 核心问答：掺到最前并重复多份——这是"从记住到理解"的锻造环节
    qa_dir = Path(args.core_qa_dir) if args.core_qa_dir else None
    core_qa: list[dict] = []
    if qa_dir and qa_dir.exists():
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
                    core_qa.append({"instruction": row["q"], "input": "",
                                    "output": row["a"]})
    if core_qa:
        data = core_qa * args.core_qa_repeat + data
        print(f"[sft] 核心问答 {len(core_qa)} 条 × {args.core_qa_repeat} 份已掺入")

    tok = ByteBPETokenizer.load(args.tokenizer)
    eos = tok.eos_id

    tok_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    used = 0
    for row in data:
        q, a = make_example(row.get("instruction", ""), row.get("input", ""),
                            row.get("output", ""))
        if not q or not a:
            continue
        q_ids = np.asarray(tok.encode(q), dtype=np.uint32)
        a_ids = np.asarray(tok.encode(a, add_eos=True), dtype=np.uint32)
        tok_parts.append(q_ids)
        tok_parts.append(a_ids)
        mask_parts.append(np.zeros(len(q_ids), dtype=np.uint8))
        mask_parts.append(np.ones(len(a_ids), dtype=np.uint8))
        used += 1

    tokens = np.concatenate(tok_parts)
    mask = np.concatenate(mask_parts)
    np.asarray(tokens, dtype=np.uint32).tofile(out_dir / "tokens.bin")
    mask.tofile(out_dir / "mask.bin")
    shutil.copy(args.tokenizer, out_dir / "tokenizer.json")
    resp_frac = mask.mean() * 100
    print(f"[sft] 用了 {used:,} 条样本 → token 流 {len(tokens):,}，"
          f"其中回答部分占 {resp_frac:.1f}%")
    print(f"[sft] {out_dir/'tokens.bin'} / {out_dir/'mask.bin'}")


if __name__ == "__main__":
    main()
