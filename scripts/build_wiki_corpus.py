# -*- coding: utf-8 -*-
"""星辰 · 语料构建：维基百科 parquet → 清洗后的 txt

清洗是轻量级的（去模板/引用/HTML/链接标记、压空白），保留正文段落；
每篇文章之间用独立行 `=DOC=` 分隔，prepare_data.py 据此切分文档并插入 eos。

用法：
  python scripts/build_wiki_corpus.py --parquet "data/wiki/raw/*.parquet" \
      --out data/wiki/corpus.txt --max_mb 300

  # Batch 03：启用 corpus-v2 的精洗引擎（默认关闭，旧行为完全不变）
  python scripts/build_wiki_corpus.py --clean --parquet "../corpus-v2/data/raw/*.parquet" \
      --out ../corpus-v2/corpus/from_build_wiki.txt
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyarrow.parquet as pq

REF_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
HEAD_RE = re.compile(r"=+\s*([^=]*?)\s*=+")
URL_RE = re.compile(r"https?://\S+")


def clean(text: str) -> str:
    t = REF_RE.sub("", text)
    t = TEMPLATE_RE.sub("", t)
    t = TEMPLATE_RE.sub("", t)          # 模板常嵌套一层，再过一遍
    t = LINK_RE.sub(r"\1", t)           # [[链接|显示]] → 显示
    t = HEAD_RE.sub(r"\1", t)           # == 标题 == → 标题
    t = TAG_RE.sub("", t)
    t = URL_RE.sub("", t)
    lines: list[str] = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_cleaner():
    """按需加载 corpus-v2 的精洗引擎；未加 --clean 时完全不触碰它。"""
    import importlib.util
    path = ROOT.parent / "corpus-v2" / "tools" / "clean_wiki.py"
    if not path.exists():
        raise FileNotFoundError(f"--clean 需要精洗引擎：{path}")
    spec = importlib.util.spec_from_file_location("clean_wiki", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "data/wiki/raw/*.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/wiki/corpus.txt"))
    ap.add_argument("--min_chars", type=int, default=150, help="短于该字符数的文章丢弃")
    ap.add_argument("--max_mb", type=float, default=300, help="输出字节数上限（0=不限）")
    ap.add_argument("--clean", action="store_true",
                    help="启用 corpus-v2 精洗引擎（参考章节跳过/乱码/简繁统一/章节切分）；默认关闭以保持旧行为")
    args = ap.parse_args()

    cleaner = load_cleaner() if args.clean else None

    files = sorted(glob.glob(args.parquet))
    if not files:
        raise FileNotFoundError(args.parquet)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kept = skipped_short = skipped_dup = 0
    bytes_written = 0
    seen_titles: set[str] = set()
    sep = "\n\n=DOC=\n\n"
    with open(out_path, "w", encoding="utf-8", buffering=1 << 20) as f:
        for fp in files:
            pf = pq.ParquetFile(fp)
            print(f"[corpus] {Path(fp).name}: {pf.metadata.num_rows:,} 行")
            for batch in pf.iter_batches(batch_size=256, columns=["title", "text"]):
                stop = False
                for title, text in zip(batch.column("title").to_pylist(),
                                       batch.column("text").to_pylist()):
                    if title in seen_titles:
                        skipped_dup += 1
                        continue
                    seen_titles.add(title)
                    if cleaner:
                        body, _reason = cleaner.clean_document(title, text)
                        if body is None:
                            skipped_short += 1
                            continue
                        docs = cleaner.split_long_doc(title, body)
                    else:
                        c = clean(text)
                        if len(c) < args.min_chars or "=DOC=" in c:
                            skipped_short += 1
                            continue
                        docs = [f"{title}\n{c}"]
                    for doc in docs:
                        if "=DOC=" in doc:
                            continue
                        f.write(doc + sep)
                        kept += 1
                        bytes_written += len(doc.encode("utf-8")) + len(sep)
                        if args.max_mb and bytes_written > args.max_mb * 1e6:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break

    dt = time.time() - t0
    print(f"[corpus] 完成：保留 {kept:,} 篇（短文丢弃 {skipped_short:,}，重名 {skipped_dup:,}），"
          f"输出 {bytes_written/1e6:.1f} MB，用时 {dt:.0f}s → {out_path}")


if __name__ == "__main__":
    main()
