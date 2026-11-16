# -*- coding: utf-8 -*-
"""星辰 · 数据管线

沿用 DeepSeek LLM 报告的预训练数据形态：整库编码为一维 token 流，
训练时随机窗口采样（memmap，不整段载入内存）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def binarize(ids: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.uint32).tofile(out_path)
    return out_path


def load_bin(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint32, mode="r")


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: torch.device):
    """随机窗口采样：x 为 [i, i+seq)，y 为右移一位的标签。

    实现要点：用**一次**花式索引取出 (B, seq+1) 的整块，再切片成 x/y。
    原先是对每行单独 `data[i:i+seq]` + `torch.stack`，会触发 batch_size 次
    Python 级切片与复制；在 4500 tok/s 时这点 CPU 开销已经能拖累吞吐。
    RNG 仍是 `torch.randint`（与旧实现完全一致，续训的随机序列不变）。

    语料比窗口还短时自动收缩窗口（小样演示 / 短验证集用）。
    """
    seq = min(seq_len, len(data) - 1)
    high = len(data) - seq          # i ∈ [0, high)，保证 y = data[i+1 : i+1+seq] 不越界
    ix = torch.randint(high, (batch_size,)).numpy()
    idx = ix[:, None] + np.arange(seq + 1, dtype=ix.dtype)[None, :]   # (B, seq+1)
    block = np.asarray(data[idx], dtype=np.int64)                     # 一次 gather
    x = torch.from_numpy(block[:, :-1].copy())
    y = torch.from_numpy(block[:, 1:].copy())
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
