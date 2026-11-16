# -*- coding: utf-8 -*-
"""多源语料 → 多份 train.bin + 分域 val.bin

为什么需要单独一个脚本：

1. `prepare_data.py` 一次只处理一个语料。训练 6 个源要手动跑 6 次，而其中 5 次都得
   带 `--init_tokenizer` 复用第一份分词器——一旦漏了就各自训出不同的分词器，
   同一串 token id 在两个 bin 里语义不同，训练出来是垃圾。
2. 更隐蔽的问题：`prepare_data.py` 的分词器样本只在**当前语料**里采。
   先跑中文维基，词表就按中文维基的分布定型；后面接英文和代码时，
   32k 个词表位置里几乎没有英文子词，压缩率会很差。
   正确做法是**从所有源按权重采样、拼在一起训一次**。

3. 每个源在末尾单独切 val，天然就是分域验证集，直接对应 `train.py` 的
   `data.val_sources`。

用法：
    python scripts/prepare_multi.py --spec configs/mix-spec.json
    python scripts/prepare_multi.py --spec ... --only zh_web,code   # 只重做某几个源
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from xingchen.tokenizer import ByteBPETokenizer        # noqa: E402
from xingchen.data import binarize                      # noqa: E402
from prepare_data import collect_corpus, read_docs, load_core_docs, encode_docs_parallel  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(src: dict) -> list[str]:
    """按 spec 条目读出文档列表。两类：目录/文件型语料，或 core（宪法+故事）。"""
    if src.get("core_dir"):
        core_dir = ROOT / src["core_dir"]
        if not core_dir.exists():
            raise FileNotFoundError(f"core 目录不存在：{core_dir}")
        base = load_core_docs(core_dir)
        repeat = int(src.get("repeat", 1))
        return base * repeat
    if not src.get("path"):
        raise ValueError(f"源 {src.get('name')} 既没有 path 也没有 core_dir")
    paths = collect_corpus(str(ROOT / src["path"]) if not Path(src["path"]).is_absolute()
                           else src["path"])
    return read_docs(paths)


def sample_for_tokenizer(docs: list[str], budget_bytes: int) -> list[str]:
    """等距抽取，不要只取开头——文件开头的格式往往和整体不一样。"""
    if budget_bytes <= 0 or not docs:
        return []
    sizes = np.fromiter((len(d.encode("utf-8")) for d in docs), dtype=np.int64, count=len(docs))
    total = int(sizes.sum())
    if total <= budget_bytes:
        return docs
    # 前缀和 + searchsorted 找等距的文档下标，保证覆盖整个文件而不是头部
    cum = np.cumsum(sizes)
    targets = np.linspace(0, total - 1, num=min(len(docs), 4000))
    idx = np.searchsorted(cum, targets)
    idx = np.unique(np.clip(idx, 0, len(docs) - 1))
    picked, used = [], 0
    for i in idx:
        if used >= budget_bytes:
            break
        picked.append(docs[int(i)])
        used += int(sizes[int(i)])
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON 规格：sources / vocab_size / out_dir")
    ap.add_argument("--only", default=None, help="只重做这些源（逗号分隔），分词器与其余 bin 不动")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = ROOT / spec.get("out_dir", "data/mix")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = int(spec.get("vocab_size", 16384))
    val_frac = float(spec.get("val_frac", 0.01))
    val_min = int(spec.get("val_min_tokens", 200_000))
    sample_mb = float(spec.get("tokenizer_sample_mb", 12))
    sources = spec["sources"]
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    if not sources:
        raise ValueError("spec 里 sources 为空")

    total_w = sum(float(s.get("weight", 1.0)) for s in sources)
    print(f"[multi] {len(sources)} 个源，合计权重 {total_w:.3f}，输出 {out_dir}")
    print(f"{'源':<14}{'权重':>8}{'目标占比':>10}")
    for s in sources:
        w = float(s.get("weight", 1.0))
        print(f"{s['name']:<14}{w:>8.3f}{w/total_w:>9.1%}")

    # ---------------- 一、分词器：从所有源按权重采样，训一次
    tok_path = out_dir / "tokenizer.json"
    need_tokenizer = not args.only or not tok_path.exists()
    if need_tokenizer:
        print(f"\n[tokenizer] 按权重从各源采样（总计约 {sample_mb:.0f} MB）")
        samples: list[str] = []
        for s in sources:
            w = float(s.get("weight", 1.0)) / total_w
            budget = int(sample_mb * 1e6 * w)
            if budget < 1000:
                print(f"  {s['name']:<14} 权重过小，跳过采样")
                continue
            t0 = time.time()
            docs = load_source(s)
            picked = sample_for_tokenizer(docs, budget)
            took = sum(len(d.encode("utf-8")) for d in picked)
            print(f"  {s['name']:<14} 采 {len(picked):>5} 篇 / {took/1e6:>5.1f} MB"
                  f"（全量 {len(docs):,} 篇，{time.time()-t0:.0f}s）")
            samples.extend(picked)

        text = "\n\n".join(samples)
        print(f"[tokenizer] 合并样本 {len(text):,} 字符")
        tok = ByteBPETokenizer(vocab_size=vocab_size)
        t0 = time.time()
        tok.train(text, max_chars=None)
        tok.save(tok_path)
        target = vocab_size - 256 - len(tok.specials)
        print(f"[tokenizer] 完成：{len(tok.merges):,}/{target:,} 条合并，{time.time()-t0:.0f}s")
        if len(tok.merges) < target * 0.92:
            print(f"[tokenizer] ⚠️ 词表只填满 {len(tok.merges)/target:.0%}：样本不足以支撑 "
                  f"{vocab_size}，多出来的位置是浪费。加大 tokenizer_sample_mb，"
                  f"或把 vocab_size 降到约 {int(len(tok.merges)*1.05)+259:,}。")
    else:
        print(f"\n[tokenizer] 复用已有 {tok_path}")
    tok = ByteBPETokenizer.load(tok_path)

    # ---------------- 二、逐源编码（一次只在内存里放一个源）
    manifest = {
        "format": "xingchen-multi-v1",
        "vocab_size": vocab_size,
        "tokenizer": {"path": str(tok_path.relative_to(ROOT)), "sha256": sha256_file(tok_path),
                      "merges": len(tok.merges)},
        "val_frac": val_frac,
        "sources": [],
    }
    eos_arr = np.array([tok.eos_id], dtype=np.uint32)
    bos_arr = np.array([tok.bos_id], dtype=np.uint32)

    print()
    for s in sources:
        name = s["name"]
        if only and name not in only:
            print(f"[{name}] 跳过（--only 未包含）")
            continue
        t0 = time.time()
        docs = load_source(s)
        raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
        arrays = encode_docs_parallel(docs, tok_path, args.workers)
        parts = [bos_arr]
        for arr in arrays:
            parts.append(arr)
            parts.append(eos_arr)
        ids = np.concatenate(parts)
        del arrays, parts, docs

        n_val = max(int(len(ids) * val_frac), val_min)
        n_val = min(n_val, len(ids) // 10)          # 最多切 10%，小源也要留够训练量
        train_ids, val_ids = ids[:-n_val], ids[-n_val:]
        train_path = out_dir / f"{name}.bin"
        val_path = out_dir / f"val_{name}.bin"
        binarize(train_ids, train_path)
        binarize(val_ids, val_path)

        rec = {
            "name": name,
            "train_bin": str(train_path.relative_to(ROOT)),
            "val_bin": str(val_path.relative_to(ROOT)),
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(len(val_ids)),
            "raw_bytes": int(raw_bytes),
            "weight": float(s.get("weight", 1.0)),
            "sha256": sha256_file(train_path),
        }
        if s.get("path"):
            rec["input"] = s["path"]
        if s.get("core_dir"):
            rec["input"] = s["core_dir"] + f" × {s.get('repeat', 1)}"
        manifest["sources"].append(rec)
        print(f"[{name}] {len(train_ids)/1e6:>7.1f}M train + {len(val_ids)/1e6:>5.2f}M val"
              f"  压缩 {raw_bytes/max(1,len(ids)):.2f} B/tok  {time.time()-t0:.0f}s")
        del ids, train_ids, val_ids

    if only:                                        # 局部重做时合并回已有 manifest
        old_path = out_dir / "manifest.json"
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            keep = {r["name"]: r for r in old.get("sources", []) if r["name"] not in only}
            merged = keep.copy()
            for r in manifest["sources"]:
                merged[r["name"]] = r
            manifest["sources"] = [merged[s["name"]] for s in sources if s["name"] in merged]

    total_train = sum(r["train_tokens"] for r in manifest["sources"])
    manifest["total_train_tokens"] = int(total_train)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[multi] 合计 {total_train/1e9:.3f}B token "
          f"（约 {total_train*3.6/1e9:.2f} GB 文本等价）")
    print(f"[multi] manifest → {out_dir/'manifest.json'}")
    print("\n训练配置片段：")
    print("data:")
    print(f"  tokenizer_path: {tok_path.relative_to(ROOT).as_posix()}")
    print("  sources:")
    for r in manifest["sources"]:
        w = r["weight"] / total_w
        print(f"    - {{path: {r['train_bin']}, weight: {w:.3f}}}")
    print("  val_sources:")
    for r in manifest["sources"]:
        print(f"    - {{path: {r['val_bin']}, name: {r['name']}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
