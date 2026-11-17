# -*- coding: utf-8 -*-
"""把 data/writings/ 的内容挑出来递给模型——她的"眼前读物"。

为什么需要它：人格内容已从预训练里摘除（不灌输），但作者每天写的东西不该石沉大海。
做法是每次对话时按话题挑最相关的几段，放进 prompt 让她"看着说话"。

这不是记忆，是读物。区别很重要：
- 读物：每次递给她，她读到，会话结束就没了
- 记忆：她自己理解后写下来的，那才是她的

检索用字符 bigram 重合度，不引入向量库：中文不需要分词，几百字的语料量
用不着 embedding，多一个依赖反而更脆。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


def _bigrams(text: str) -> set[str]:
    """字符二元组。中文按字切，能抓住"星星""灯塔"这类词，不需要分词器。"""
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}


class PersonaReader:
    """读取作者写的内容，按相关度为当前话题挑段落。

    `max_chars` 是塞进 prompt 的上限。166M 的 seq_len 只有 512~4096，
    必须给对话本身留空间——超过这个数就会把问题挤出窗口，得不偿失。
    """

    def __init__(self, root: Path, max_chars: int = 380, min_chunk: int = 12):
        self.root = Path(root)
        self.max_chars = max_chars
        self.min_chunk = min_chunk
        self.chunks: list[dict] = []
        self._load()
        self._build_index()

    def _load(self) -> None:
        if not self.root.exists():
            return
        const = self.root / "constitution.txt"
        if const.exists():
            for para in self._split(const.read_text(encoding="utf-8", errors="ignore")):
                self.chunks.append({"src": "宪法", "text": para})

        stories = self.root / "stories"
        if stories.exists():
            for f in sorted(stories.glob("*.txt")):
                for para in self._split(f.read_text(encoding="utf-8", errors="ignore")):
                    self.chunks.append({"src": f.stem, "text": para})

        qa_dir = self.root / "qa"
        if qa_dir.exists():
            for f in sorted(qa_dir.glob("*.jsonl")):
                for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    q, a = row.get("q") or row.get("question"), row.get("a") or row.get("answer")
                    if q and a:
                        self.chunks.append({"src": f"问答/{f.stem}", "text": f"问：{q}\n答：{a}"})

    def _split(self, text: str) -> list[str]:
        """按空行切段，过长再按句号切。太短的段落跳过（没有信息量）。"""
        out = []
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if len(block) < self.min_chunk:
                continue
            if len(block) <= self.max_chars:
                out.append(block)
                continue
            buf = ""
            for sent in re.split(r"(?<=[。！？\n])", block):
                if len(buf) + len(sent) > self.max_chars and buf:
                    out.append(buf.strip())
                    buf = sent
                else:
                    buf += sent
            if len(buf.strip()) >= self.min_chunk:
                out.append(buf.strip())
        return out

    def __len__(self) -> int:
        return len(self.chunks)

    def _build_index(self) -> None:
        """预计算每个 bigram 出现在多少段里，用于 IDF 加权。

        没有这层加权时，"记得""什么"这类高频词会盖过真正的关键词：
        问"你记得灯塔吗"会被判到"晚安"那段，因为两句共用"记得"。
        IDF 让稀有词（灯塔）主导打分，常见词自动降权。
        """
        self._grams: list[set[str]] = []
        df: dict[str, int] = {}
        for c in self.chunks:
            g = _bigrams(c["text"])
            self._grams.append(g)
            for x in g:
                df[x] = df.get(x, 0) + 1
        self._idf = {x: math.log((len(self.chunks) + 1) / (n + 0.5)) for x, n in df.items()}

    def relevant(self, query: str, k: int = 2, budget: int | None = None) -> str:
        """挑 k 段最相关的，拼起来不超过 budget 字符。没有相关内容时返回空串。"""
        if not self.chunks or not query.strip():
            return ""
        budget = budget or self.max_chars
        qg = _bigrams(query)
        scored = []
        for i, c in enumerate(self.chunks):
            shared = qg & self._grams[i]
            if not shared:
                continue
            # IDF 加权：稀有词得分高，长段落略降权避免挤掉短而准的
            score = sum(self._idf.get(x, 0.0) for x in shared)
            scored.append((score / (1 + len(c["text"]) / budget), c))
        if not scored:
            return ""
        scored.sort(key=lambda x: -x[0])

        picked, total = [], 0
        for _, c in scored[:max(k, 1)]:
            if total + len(c["text"]) > budget:
                continue
            picked.append(c)
            total += len(c["text"])
            if len(picked) >= k:
                break
        if not picked:
            return ""
        return "\n\n".join(c["text"] for c in picked)

    def explain(self, query: str, k: int = 2, budget: int | None = None) -> list[dict]:
        """和 relevant 同样的挑选逻辑，但返回来源信息，用来打印"她读到了什么"。"""
        picked_text = self.relevant(query, k, budget)
        if not picked_text:
            return []
        return [{"src": c["src"], "chars": len(c["text"]), "preview": c["text"][:40]}
                for c in self.chunks if c["text"] in picked_text]
