# -*- coding: utf-8 -*-
"""BPE 合并引擎的 C 实现包装（ctypes 调用同目录的 fastbpe.dll）。

为什么只换这一层：tokenizer 里真正耗 CPU 的是「每轮扫一遍相邻对、找最小 rank、
再整体合并」这个纯 Python 循环（编码 300 MB 语料时要跑上亿次）。
C 版保持逐字节可复现的相同语义，其余部分（正则预切分、特殊 token、词表存取）仍在 Python。

设计成"可选加速"：
  * DLL 缺失 / 架构不匹配 / 加载失败 → available() 返回 False；
  * tokenizer 检测到不可用时自动退回原来的纯 Python 路径，行为完全不变。
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence, Tuple

_DLL_PATH = Path(__file__).with_name("fastbpe.dll")

_lib = None
_load_error: Exception | None = None
_load_tried = False


def _load():
    global _lib, _load_error, _load_tried
    if _load_tried:
        return _lib
    _load_tried = True
    try:
        lib = ctypes.CDLL(str(_DLL_PATH))
        lib.xc_bpe_create.restype = ctypes.c_void_p
        lib.xc_bpe_create.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.xc_bpe_destroy.restype = None
        lib.xc_bpe_destroy.argtypes = [ctypes.c_void_p]
        lib.xc_bpe_encode.restype = ctypes.c_int32
        lib.xc_bpe_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        _lib = lib
    except OSError as exc:            # 文件不存在、位数不匹配、缺依赖
        _load_error = exc
        _lib = None
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> Exception | None:
    _load()
    return _load_error


class FastBPE:
    """把 tokenizer 的 merges 表交给 C，之后逐片段编码。"""

    __slots__ = ("_lib", "_h", "_buf", "_cap")

    def __init__(self, merges: Sequence[Tuple[int, int]]):
        lib = _load()
        if lib is None:
            raise RuntimeError(f"fastbpe.dll 不可用：{_load_error}")
        n = len(merges)
        if n <= 0:
            raise RuntimeError("merges 为空")
        arr_a = (ctypes.c_int32 * n)(*[int(p[0]) for p in merges])
        arr_b = (ctypes.c_int32 * n)(*[int(p[1]) for p in merges])
        arr_r = (ctypes.c_int32 * n)(*range(n))
        handle = lib.xc_bpe_create(arr_a, arr_b, arr_r, n)
        if not handle:
            raise RuntimeError("xc_bpe_create 返回空（内存不足？）")
        self._lib = lib
        self._h = handle
        self._cap = 4096
        self._buf = (ctypes.c_int32 * self._cap)()

    def encode_piece(self, piece: str) -> List[int]:
        data = piece.encode("utf-8")
        n = len(data)
        if n == 0:
            return []
        if n > self._cap:
            self._cap = n
            self._buf = (ctypes.c_int32 * n)()
        got = self._lib.xc_bpe_encode(self._h, data, n, self._buf, self._cap)
        return self._buf[:got]

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.xc_bpe_destroy(self._h)
            self._h = None

    def __del__(self):                      # pragma: no cover - 解释器退出时尽力释放
        try:
            self.close()
        except Exception:
            pass
