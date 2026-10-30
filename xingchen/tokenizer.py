# -*- coding: utf-8 -*-
"""星辰 · 从零实现的字节级 BPE 分词器（支持 32k 词表、百 MB 级语料）

两个关键工程点（都是 GPT-2 时代的标准做法，此处手写）：
1. 预切分：先用正则把文本切成短片段（连续汉字≤6 字、英文单词、数字串、空白串、
   单个标点），BPE 合并只在片段内部进行。把"对全文反复扫描"拆成"对海量短片段
   各自快速合并"，编码速度提升几个数量级。
2. 增量训练：用"堆 + 局部计数"代替每轮全量重建——每次合并只重算受影响的片段，
   32k 词表也能在几分钟内训完。

词表布局：[0, 256) 字节节；[256, 256+n_merges) 合并节；末尾为特殊 token 节。
"""
from __future__ import annotations

import base64
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>"]

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
    def train(self, text: str, verbose: bool = True, log_every: int = 2048) -> None:
        """在给定文本样本上训练合并规则。样本取语料的代表性切片即可（几 MB～几十 MB）。"""
        target_merges = self.vocab_size - 256 - len(SPECIALS)
        pieces: List[List[int]] = [list(p.encode("utf-8")) for p in _PIECE_RE.findall(text)]
        if verbose:
            print(f"[tokenizer] 样本 {len(text):,} 字符 → {len(pieces):,} 个片段")

        counts: dict = defaultdict(int)        # pair -> 当前全库出现次数
        locs: dict = defaultdict(set)          # pair -> 涉及哪些片段（下标集合）
        for i, piece in enumerate(pieces):
            for pr in zip(piece, piece[1:]):
                counts[pr] += 1
                locs[pr].add(i)
        heap = [(-c, pr) for pr, c in counts.items()]
        heapq.heapify(heap)

        merges: List[tuple] = []
        while len(merges) < target_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            cur = counts.get(pair, 0)
            if cur != -neg_c:                  # 过期条目：按当前真实计数放回
                if cur > 0:
                    heapq.heappush(heap, (-cur, pair))
                continue
            if cur < 2:                        # 堆顶已不足 2 次，全表都不足 2
                break
            a, b = pair
            new_id = 256 + len(merges)
            affected, locs[pair] = locs[pair], set()
            counts[pair] = 0                   # 该 pair 已被整体替换，永不再出现
            for i in affected:
                old = pieces[i]
                for pr in zip(old, old[1:]):   # 撤掉旧片段的相邻对计数
                    counts[pr] -= 1
                new_piece: List[int] = []
                j, n = 0, len(old)
                while j < n:                   # 片段内从左到右非重叠合并
                    if j < n - 1 and old[j] == a and old[j + 1] == b:
                        new_piece.append(new_id)
                        j += 2
                    else:
                        new_piece.append(old[j])
                        j += 1
                for pr in zip(new_piece, new_piece[1:]):   # 登记新相邻对
                    counts[pr] += 1
                    locs[pr].add(i)
                    heapq.heappush(heap, (-counts[pr], pr))
                pieces[i] = new_piece
            merges.append(pair)
            if verbose and len(merges) % log_every == 0:
                print(f"[tokenizer] merge {len(merges):,}/{target_merges:,}")

        self.merges = merges
        self._rebuild_tables()
        if verbose:
            print(f"[tokenizer] 训练完成：{len(merges):,} 条合并规则")

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
