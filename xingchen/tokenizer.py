# -*- coding: utf-8 -*-
"""星辰 · 从零实现的字节级 BPE 分词器（支持 32k 词表、百 MB 级语料）

两个关键工程点（都是 GPT-2 时代的标准做法，此处手写）：
1. 预切分：先用正则把文本切成短片段（连续汉字≤6 字、英文单词、数字串、空白串、
   单个标点），BPE 合并只在片段内部进行。把"对全文反复扫描"拆成"对海量短片段
   各自快速合并"，编码速度提升几个数量级。
2. **向量化训练**：把全部片段拼成一条 uint32 数组（片段之间插分隔符），每轮
   「统计相邻对 → 取最高频 → 非重叠合并」全部用 numpy 完成。原先是纯 Python 的
   堆 + 字典增量更新，16k 词表跑几小时；改成向量化后单轮 15ms/百万 token，
   16k 约 3 分钟、32k 约 6 分钟。

性能要点（改动时别踩回去）：
- pair 打包成 `(left << 16) | right` 的 uint32，而不是 Python tuple 做 dict key。
  前提是 id < 65536，见 `_SEP` 处的断言。
- 片段分隔符用哨兵值 `_SEP = 0xFFFF`，**不做掩码过滤**——只把含哨兵的 pair 计数置 0。
  用布尔掩码切 `key[valid]` 会多一次全量复制，实测慢 3 倍。
- 合并时用「写入左端 + 布尔索引删右端」，**不要用 argsort 重排**：布尔索引是 O(n)
  顺序扫描，argsort 是 O(n log n) 且缓存不友好，后者慢 4 倍。

词表布局：[0, 256) 字节节；[256, 256+n_merges) 合并节；末尾为特殊 token 节。
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>"]

# 片段分隔符：一个不可能作为真实 id 的值（词表上限 65536，见 train 里的断言）
_SEP = 0xFFFF
_U16 = np.uint32(16)
_U16MASK = np.uint32(0xFFFF)

# 预切分规则（按顺序优先匹配）：CJK 连续段（限长 6）| 英文字母串 | 数字串 | 空白串 | 其他单字符
_PIECE_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{1,6}"
    r"|[a-zA-Z]+"
    r"|[0-9]+"
    r"|\s+"
    r"|."
)


class ByteBPETokenizer:
    def __init__(self, vocab_size: int = 4096):
        assert vocab_size >= 256 + len(SPECIALS)
        self.vocab_size = vocab_size
        self.merges: List[tuple] = []          # [(id_a, id_b), ...] 按学到的顺序
        self.specials: dict = {}
        self._pair_rank: dict = {}
        self._id_to_bytes: List[bytes] = []
        self._fast = None                      # C 加速引擎（可用时才非 None）

    # ------------------------------------------------------------------ 训练
    def train(self, text: str, verbose: bool = True, log_every: int = 2048,
              max_chars: int | None = None) -> None:
        """在给定文本样本上训练合并规则（numpy 向量化，32k 词表约 6 分钟）。

        `max_chars`：训练样本上限。合并计数只关心**频率分布**，等距抽样不会改变分布形状，
        但能把训练时间压到线性级——16k 合并用几十 MB 是浪费，一两百万 token 就够。
        32k 词表建议 600000~1200000 字符。
        """
        if max_chars and len(text) > max_chars:
            seg = 2000
            total = len(text) // seg
            keep = max(1, max_chars // seg)
            stride = max(1, total // keep)
            text = "".join(text[i * seg:(i + 1) * seg] for i in range(0, total, stride))
            if verbose:
                print(f"[tokenizer] 样本超限，等距抽样至 {len(text):,} 字符（stride={stride}）")

        target_merges = self.vocab_size - 256 - len(SPECIALS)
        # 打包用 16 位存 id，所以 id 必须 < _SEP
        assert 256 + target_merges + len(SPECIALS) <= _SEP, (
            f"词表 {self.vocab_size} 超出向量化训练上限（≤{_SEP}）")

        pieces = [p.encode("utf-8") for p in _PIECE_RE.findall(text)]
        if not pieces:
            raise ValueError("训练文本为空")
        arr = self._pack_pieces(pieces)
        if verbose:
            print(f"[tokenizer] 样本 {len(text):,} 字符 → {len(pieces):,} 个片段 / "
                  f"{len(arr):,} 个初始 token")

        merges: List[tuple] = []
        while len(merges) < target_merges:
            top, freq = self._most_common_pair(arr)
            if top is None or freq < 2:     # 全库已无出现 2 次以上的相邻对
                break
            new_id = 256 + len(merges)
            merges.append((top >> 16, top & 0xFFFF))
            arr = self._merge_pair(arr, top, new_id)
            if verbose and len(merges) % log_every == 0:
                print(f"[tokenizer] merge {len(merges):,}/{target_merges:,}  "
                      f"freq={freq:,}  token 数={len(arr):,}")

        self.merges = merges
        self._rebuild_tables()
        if verbose:
            print(f"[tokenizer] 训练完成：{len(merges):,} 条合并规则，"
                  f"语料压缩到 {len(arr):,} token")

    # ------------------------------------------------------------------ 训练用的向量化内核
    @staticmethod
    def _pack_pieces(pieces: List[bytes]) -> np.ndarray:
        """把片段列表拼成一条 uint32 数组，片段之间插入 `_SEP`。

        分隔符保证 BPE 合并不会跨片段边界——否则「句子末尾的字 + 下句开头」会
        被误学成一个 token。这是预切分方案的正确性前提，不能省。
        """
        lens = np.fromiter((len(p) for p in pieces), dtype=np.int64, count=len(pieces))
        ends = np.cumsum(lens)
        total = int(ends[-1]) + len(pieces)
        arr = np.full(total, _SEP, dtype=np.uint32)
        valid = np.ones(total, dtype=bool)
        valid[ends + np.arange(len(pieces))] = False     # 分隔符占位不参与
        arr[valid] = np.frombuffer(b"".join(pieces), dtype=np.uint8)
        return arr

    @staticmethod
    def _most_common_pair(arr: np.ndarray) -> tuple[Optional[int], int]:
        """统计相邻对，返回 (打包后的 pair, 出现次数)。O(n log n)（unique 内部排序）。"""
        if len(arr) < 2:
            return None, 0
        key = (arr[:-1] << _U16) | arr[1:]
        uniq, cnt = np.unique(key, return_counts=True)
        # 含分隔符的对不是真实相邻对，计数置 0（比布尔掩码过滤快，见模块 docstring）
        cnt[(uniq >> _U16) == _SEP] = 0
        cnt[(uniq & _U16MASK) == _SEP] = 0
        k = int(np.argmax(cnt))
        if cnt[k] <= 0:
            return None, 0
        return int(uniq[k]), int(cnt[k])

    @staticmethod
    def _merge_pair(arr: np.ndarray, top: int, new_id: int) -> np.ndarray:
        """把 arr 里所有 top 出现的位置非重叠地替换成 new_id（从左到右，同 BPE 语义）。"""
        key = (arr[:-1] << _U16) | arr[1:]
        hit = np.nonzero(key == top)[0]
        if len(hit) == 0:
            return arr
        # 相邻命中会共享一个元素（如 a a a 里的两个 (a,a)），只保留左边那个
        take = np.ones(len(hit), dtype=bool)
        take[1:] = hit[1:] > hit[:-1] + 1
        pos = hit[take]
        out = arr.copy()
        out[pos] = new_id
        keep = np.ones(len(arr), dtype=bool)
        keep[pos + 1] = False
        return out[keep]

    @staticmethod
    def _merge_sequence(ids: List[int], pair: tuple, new_id: int) -> List[int]:
        out = []
        i = 0
        a, b = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _rebuild_tables(self) -> None:
        self._id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._id_to_bytes.append(self._id_to_bytes[a] + self._id_to_bytes[b])
        for i, name in enumerate(SPECIALS):
            self.specials[name] = 256 + len(self.merges) + i
        self._pair_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self._init_fast()

    def _init_fast(self) -> None:
        """有 C 加速引擎（xingchen/fastbpe.dll）就用它做片段合并；没有则保持纯 Python。"""
        self._fast = None
        try:
            from .fastbpe import FastBPE, available
        except Exception:
            return
        if not available() or not self.merges:
            return
        try:
            self._fast = FastBPE(self.merges)
        except Exception:
            self._fast = None

    # ------------------------------------------------------------------ 编解码
    def _encode_piece(self, piece: str) -> List[int]:
        if self._fast is not None:
            return self._fast.encode_piece(piece)
        cur = list(piece.encode("utf-8"))
        rank = self._pair_rank
        while len(cur) >= 2:
            best_rank = None
            best_pair = None
            for pr in zip(cur, cur[1:]):
                r = rank.get(pr)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pr
            if best_rank is None:
                break
            cur = self._merge_sequence(cur, best_pair, 256 + best_rank)
        return cur

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        for piece in _PIECE_RE.findall(text):
            ids.extend(self._encode_piece(piece))
        if add_bos:
            ids = [self.specials["<|bos|>"]] + ids
        if add_eos:
            ids = ids + [self.specials["<|eos|>"]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = bytearray()
        n_special_start = 256 + len(self.merges)
        for i in ids:
            if 0 <= i < n_special_start:
                data += self._id_to_bytes[i]
            # 特殊 token 不参与文本还原
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> None:
        payload = {
            "format": "xingchen-bpe-v1",
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
            "specials": self.specials,
            # 词表条目以 base64 存原始字节，避免编码问题
            "vocab_b64": [base64.b64encode(tok).decode("ascii")
                          for tok in self._id_to_bytes],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(payload["vocab_size"])
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok.specials = dict(payload["specials"])
        tok._id_to_bytes = [base64.b64decode(s) for s in payload["vocab_b64"]]
        tok._pair_rank = {pair: rank for rank, pair in enumerate(tok.merges)}
        tok._init_fast()
        return tok

    @property
    def bos_id(self) -> int:
        return self.specials["<|bos|>"]

    @property
    def eos_id(self) -> Optional[int]:
        return self.specials["<|eos|>"]
