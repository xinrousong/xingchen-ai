# -*- coding: utf-8 -*-
"""星辰 · 数据准备：训练 BPE 分词器并把语料编码为 token 二进制

用法：
  python scripts/prepare_data.py --corpus data/wiki/corpus.txt \
      --vocab_size 16384 --out_dir data/wiki

corpus 可以是单个 txt、目录（递归收集 *.txt）或逗号分隔的多个路径。
文本中若出现独立行 `=DOC=`，会按它切分文档，文档之间写入 eos。
编码用多进程并行（Windows 下同样可用），失败时自动退回单进程。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from xingchen.tokenizer import ByteBPETokenizer
from xingchen.data import binarize

DOC_SEP = "\n\n=DOC=\n\n"


def collect_corpus(corpus: str) -> list[Path]:
    paths: list[Path] = []
    for part in corpus.split(","):
        p = Path(part.strip())
        if p.is_dir():
            paths += sorted(p.rglob("*.txt"))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"找不到语料：{p}")
    return paths


def read_docs(paths: list[Path]) -> list[str]:
    docs: list[str] = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for d in text.split(DOC_SEP):
            d = d.strip()
            if d:
                docs.append(d)
    return docs


def load_core_docs(core_dir: Path) -> list[str]:
    """核心记忆 = 宪法 + stories/ 下所有故事。作为常驻文档掺入语料。"""
    docs: list[str] = []
    const = core_dir / "constitution.txt"
    if const.exists() and const.read_text(encoding="utf-8").strip():
        docs.append(const.read_text(encoding="utf-8").strip())
    stories = core_dir / "stories"
    if stories.exists():
        for p in sorted(stories.glob("*.txt")):
            t = p.read_text(encoding="utf-8").strip()
            if t:
                docs.append(t)
    return docs


# ---------------- 多进程编码 worker（Windows spawn 需要模块级函数） ----------------
_TOK: ByteBPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _TOK
    _TOK = ByteBPETokenizer.load(tok_path)


def _encode_docs(docs: list[str]) -> list[np.ndarray]:
    return [np.asarray(_TOK.encode(d), dtype=np.uint32) for d in docs]


def encode_docs_parallel(docs: list[str], tok_path: Path, workers: int) -> list[np.ndarray]:
    chunks = [docs[i:i + 512] for i in range(0, len(docs), 512)]
    arrays: list[np.ndarray] = []
    t0 = time.time()
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, mp.cpu_count()), initializer=_init_worker,
                      initargs=(str(tok_path),)) as pool:
            for k, part in enumerate(pool.imap_unordered(_encode_docs, chunks)):
                arrays.extend(part)
                done = len(arrays)
                if done % 25000 < 512 or done == len(docs):
                    print(f"[prepare] 已编码 {done:,}/{len(docs):,} 篇 "
                          f"({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"[prepare] 并行编码失败（{e}），退回单进程")
        tok = ByteBPETokenizer.load(tok_path)
        arrays = [np.asarray(tok.encode(d), dtype=np.uint32) for d in docs]
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--sample_mb", type=float, default=8.0,
                    help="用于训练分词器的文本量上限（MB，取语料开头切片）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--val_min_tokens", type=int, default=4096)
    ap.add_argument("--init_tokenizer", default=None,
                    help="复用已有分词器文件（与旧检查点词表保持一致），跳过分词器训练")
    ap.add_argument("--core_dir", default=str(ROOT / "data/core"),
                    help="核心记忆目录（宪法+故事），会掺入语料；传空字符串可关闭")
    ap.add_argument("--core_repeat", type=int, default=20,
                    help="核心记忆在语料中大约重复的份数（记忆强度，越大越牢固）")
    ap.add_argument("--core_qa_dir", default=str(ROOT / "data/core/qa"),
                    help="核心问答目录（用于生成课文+问答混合的巩固集）")
    args = ap.parse_args()

    paths = collect_corpus(args.corpus)
    print(f"[prepare] 语料文件 {len(paths)} 个")
    docs = read_docs(paths)

    core_dir = Path(args.core_dir) if args.core_dir else None
    core_docs = load_core_docs(core_dir) if core_dir else []
    if core_docs:
        interval = max(1, len(docs) // max(1, args.core_repeat))
        woven: list[str] = []
        for i, d in enumerate(docs):
            if i % interval == 0:
                woven.extend(core_docs)
            woven.append(d)
        woven.extend(core_docs)          # 结尾再放一组，保证首尾都有记忆
        docs = woven
        print(f"[prepare] 已掺入核心记忆 {len(core_docs)} 篇 × 约 {args.core_repeat} 份"
              f"（每 {interval} 篇插一组）")

    raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"[prepare] 文档 {len(docs):,} 篇 / {raw_bytes/1e6:.1f} MB")

    # 分词器训练样本：按文档累加到 sample_mb 为止
    sample: list[str] = []
    sample_bytes = 0
    for d in docs:
        sample.append(d)
        sample_bytes += len(d.encode("utf-8"))
        if sample_bytes >= args.sample_mb * 1e6:
            break
    print(f"[prepare] 分词器训练样本 {len(sample):,} 篇 / {sample_bytes/1e6:.1f} MB")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    if args.init_tokenizer:
        import shutil
        tok = ByteBPETokenizer.load(args.init_tokenizer)
        if tok.vocab_size != args.vocab_size:
            raise ValueError(f"复用的分词器 vocab={tok.vocab_size} 与 --vocab_size={args.vocab_size} 不一致")
        src = Path(args.init_tokenizer)
        if src.resolve() != tok_path.resolve():     # 同一文件则无需拷贝
            shutil.copy(src, tok_path)
        print(f"[prepare] 复用分词器：{tok_path}（合并 {len(tok.merges):,} 条）")
    else:
        tok = ByteBPETokenizer(vocab_size=args.vocab_size)
        tok.train("\n\n".join(sample))
        tok.save(tok_path)
        print(f"[prepare] 分词器已保存：{tok_path}（词表 {args.vocab_size}，合并 {len(tok.merges):,} 条）")

    arrays = encode_docs_parallel(docs, tok_path, args.workers)
    eos = np.array([tok.eos_id], dtype=np.uint32)
    bos = np.array([tok.bos_id], dtype=np.uint32)
    parts = [bos] + [a for arr in arrays for a in (arr, eos)]
    ids = np.concatenate(parts)
    del arrays, parts
    print(f"[prepare] 总 token {len(ids):,}（压缩比 {raw_bytes/len(ids):.2f} 字节/token）")

    n_val = max(int(len(ids) * args.val_frac), args.val_min_tokens)
    n_val = min(n_val, len(ids) // 10)
    train_path = binarize(ids[:-n_val], out_dir / "train.bin")
    val_path = binarize(ids[-n_val:], out_dir / "val.bin")
    print(f"[prepare] 训练 {len(ids)-n_val:,} / 验证 {n_val:,} token")
    print(f"[prepare] {train_path} / {val_path}")

    # 记忆巩固集：核心课文 + 核心问答（问：…答：格式）交错堆叠成约 30 万 token。
    # 只有裸课文会把模型练成"背书机"（会背但不应题），必须混合问答才能两头都保住。
    if core_docs:
        core_ids: list[int] = []
        for d in core_docs:
            core_ids.extend(tok.encode(d))
            core_ids.append(tok.eos_id)
        qa_dir = Path(args.core_qa_dir) if args.core_qa_dir else None
        if qa_dir and Path(qa_dir).exists():
            import json as _json
            for p in sorted(Path(qa_dir).glob("*.jsonl")):
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if row.get("q") and row.get("a"):
                        core_ids.extend(tok.encode(f"问：{row['q']}\n答：{row['a']}"))
                        core_ids.append(tok.eos_id)
        core_arr = np.asarray(core_ids, dtype=np.uint32)

        # 巩固集逐遍洗牌：同一段材料重复上百遍时，若顺序固定，模型会背"循环的顺序"
        # 而不是"问题→答案"的映射；每遍随机打乱才能练出真正的条件映射
        import random as _random
        segments = core_ids[:]                      # 一整组（课文+问答，eos 分隔）
        seg_bounds = []
        pos = 0
        for d in core_docs:
            n = len(tok.encode(d)) + 1
            seg_bounds.append((pos, pos + n)); pos += n
        chunks: list[np.ndarray] = []
        made = 0
        rep = 0
        while made < 300_000:
            rep += 1
            order = list(range(len(seg_bounds)))
            _random.Random(rep).shuffle(order)
            for idx in order:
                a, b = seg_bounds[idx]
                chunks.append(core_arr[a:b])
                made += b - a
        np.concatenate(chunks).tofile(out_dir / "core_topup.bin")
        print(f"[prepare] 记忆巩固集 core_topup.bin：{made:,} token × {rep} 遍"
              f"（{len(seg_bounds)} 段课文+问答，逐遍洗牌）")


if __name__ == "__main__":
    main()
