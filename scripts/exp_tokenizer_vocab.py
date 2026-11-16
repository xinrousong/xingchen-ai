# -*- coding: utf-8 -*-
"""分词器词表规模对比实验（决定 32k 值不值得换）

背景：现有分词器 16k，中文约 0.9 token/字，属于偏低效。换 32k 能省多少 token、
多花多少参数，必须用**同一份语料实测**，不能拍脑袋。

做法：取语料前 N MB 训分词器，再用后面一段**不重叠**的文本测压缩率。

用法：
    python scripts/exp_tokenizer_vocab.py \
        --corpus ../corpus-v2/corpus/pretrain_wiki_clean.txt \
        --train-mb 6 --test-mb 1 --vocabs 16384,24576,32768
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xingchen.tokenizer import ByteBPETokenizer  # noqa: E402


def read_prefix(path: Path, max_bytes: int) -> str:
    """按字节读前缀，末尾按字符边界截断（避免半个汉字）。"""
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    return raw.decode("utf-8", errors="ignore")[:-1]


def read_slice(path: Path, start: int, length: int) -> str:
    """从 start 字节处读 length 字节，用于取不相交的测试段。"""
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(length)
    return raw.decode("utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--train-mb", type=float, default=6.0)
    ap.add_argument("--test-mb", type=float, default=1.0)
    ap.add_argument("--vocabs", default="16384,32768")
    ap.add_argument("--dim", type=int, default=768, help="用于估算 embedding 参数量")
    ap.add_argument("--max-train-chars", type=int, default=400_000,
                    help="训练样本上限（等距抽样）；纯 Python 训练下这个上限决定耗时")
    args = ap.parse_args()

    path = Path(args.corpus)
    if not path.exists():
        print(f"[FATAL] 找不到语料 {path}")
        return 2

    train_bytes = int(args.train_mb * 1024 * 1024)
    test_bytes = int(args.test_mb * 1024 * 1024)

    train_text = read_prefix(path, train_bytes)
    test_text = read_slice(path, train_bytes, test_bytes)     # 与训练段不重叠
    tb = len(train_text.encode("utf-8"))
    eb = len(test_text.encode("utf-8"))
    chars = sum(1 for ch in test_text if "\u4e00" <= ch <= "\u9fff")
    print(f"训练段 {tb/1e6:.1f} MB / {len(train_text):,} 字符")
    print(f"测试段 {eb/1e6:.1f} MB / {len(test_text):,} 字符（汉字 {chars:,}）\n")

    print(f"{'词表':>8} {'合并条数':>10} {'测试 token':>12} {'token/汉字':>10} "
          f"{'汉字/token':>10} {'训练耗时':>10} {'embedding':>10}")
    rows = []
    for vs in [int(x) for x in args.vocabs.split(",")]:
        t0 = time.time()
        tok = ByteBPETokenizer(vocab_size=vs)
        tok.train(train_text, verbose=False, max_chars=args.max_train_chars)
        train_s = time.time() - t0
        ids = tok.encode(test_text)
        n_tok = len(ids)
        per_char = n_tok / max(1, chars)
        char_per_tok = chars / max(1, n_tok)
        emb = vs * args.dim / 1e6
        print(f"{vs:>8} {len(tok.merges):>10,} {n_tok:>12,} {per_char:>10.3f} "
              f"{char_per_tok:>10.3f} {train_s:>9.0f}s {emb:>9.2f}M")
        rows.append((vs, n_tok, per_char, len(tok.merges)))

    if len(rows) >= 2:
        base_vs, base_n, base_pc, _ = rows[0]
        print()
        for vs, n, pc, _ in rows[1:]:
            saved = (1 - n / base_n) * 100
            print(f"{vs} 相对 {base_vs}：token 数少 {saved:.1f}%，"
                  f"同样算力可多看 {saved/(100-saved)*100:.1f}% 的文本；"
                  f"embedding 多 {(vs-base_vs)*args.dim/1e6:.1f}M 参数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
