# -*- coding: utf-8 -*-
"""校验 C 版 BPE 与纯 Python 版逐 token 一致，并测速。只读，不改任何数据。"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xingchen.tokenizer import ByteBPETokenizer  # noqa: E402

rep = []
def out(s=""):
    rep.append(str(s))

tok = ByteBPETokenizer.load(ROOT / "data" / "wiki" / "tokenizer.json")
out(f"fast 引擎: {'可用' if tok._fast is not None else '不可用（会退回纯 Python）'}")
if tok._fast is None:
    from xingchen.fastbpe import load_error
    out(f"加载错误: {load_error()}")

# ---------- 一致性 ----------
corpus = ROOT / "data" / "wiki" / "corpus.txt"
raw = corpus.read_text(encoding="utf-8", errors="replace")
out(f"语料 {len(raw):,} 字符")

random.seed(0)
samples = []
for _ in range(300):
    s = random.randrange(0, max(1, len(raw) - 400))
    samples.append(raw[s:s + random.randrange(1, 400)])
samples += ["", "a", "，", "😀🚀", "Hello, world!", "中文English混合123", " " * 37, "\n\n\n"]

fast_ids = [tok.encode(t) for t in samples]
tok._fast = None
slow_ids = [tok.encode(t) for t in samples]
tok._init_fast()

bad = [i for i, (a, b) in enumerate(zip(fast_ids, slow_ids)) if a != b]
out(f"一致性：{len(samples)-len(bad)}/{len(samples)} 段完全相同" + (f"，不一致下标 {bad[:5]}" if bad else " ✅"))
total_tok = sum(len(x) for x in fast_ids)
out(f"共比对 {total_tok:,} 个 token")

# ---------- 测速 ----------
block = raw[: 4_000_000]
t0 = time.perf_counter(); ids_a = tok.encode(block); t_fast = time.perf_counter() - t0
tok._fast = None
t0 = time.perf_counter(); ids_b = tok.encode(block); t_slow = time.perf_counter() - t0
tok._init_fast()
out("")
out(f"编码 400 万字符（{len(ids_a):,} token）")
out(f"  纯 Python : {t_slow:6.2f} s")
out(f"  C 引擎    : {t_fast:6.2f} s   →  加速 {t_slow/t_fast:.1f}×")
out(f"  结果一致  : {'是' if ids_a == ids_b else '否 ❌'}")

Path(ROOT / "logs" / "fastbpe_check.txt").write_text("\n".join(rep), encoding="utf-8")
print("written logs/fastbpe_check.txt")
