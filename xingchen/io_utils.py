# -*- coding: utf-8 -*-
"""文件写入的原子操作（带 Windows 文件锁重试）。"""

from __future__ import annotations

import os
import time
from pathlib import Path


def atomic_replace(src: Path | str, dst: Path | str, tries: int = 8) -> None:
    """把 src 原子替换到 dst，对 Windows 的短暂文件锁自动重试。

    为什么需要重试：Windows 上 os.replace 底层是 MoveFileEx(REPLACE_EXISTING)，
    只要目标文件被任何进程以不共享的方式打开，就返回 ERROR_ACCESS_DENIED（WinError 5）。
    实测在 torch.save 刚写出一个 2GB 的 .pt 之后立刻 replace，会被 Defender 的实时
    扫描锁住——频繁存档的长跑训练每次都可能踩到。POSIX 上不存在这个问题。

    退避序列 0.4+0.8+1.6+3.2+6.4+12+12+12 约 48 秒，足够杀毒扫完一个大文件；
    超过仍然失败就抛出，由调用方决定。
    """
    src, dst = Path(src), Path(dst)
    delay = 0.4
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 12.0)


def safe_unlink(path: Path | str) -> None:
    """删临时文件，失败不抛（它只是清理，不该掩盖真正的错误）。"""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
