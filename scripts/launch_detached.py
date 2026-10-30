# -*- coding: utf-8 -*-
"""把任意命令以「无控制台、独立进程」的方式拉起来，然后立刻返回。

用途：长跑任务（训练链、评测）不该跟着终端窗口或编辑器一起死。
注意：如果父进程处在一个不允许 breakaway 的作业对象里，子进程仍会留在那个作业中；
这时只有“由资源管理器启动”才能真正脱离（见 start_all.cmd 的注释）。

用法：
    python scripts/launch_detached.py scripts/m15_chain.cmd
    python scripts/launch_detached.py --log logs/m15.log -- <任意命令与参数>
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    log_name = None
    if argv[0] == "--log":
        log_name = argv[1]
        argv = argv[2:]
        if argv and argv[0] == "--":
            argv = argv[1:]
    if not argv:
        print("没有要执行的命令", file=sys.stderr)
        return 2

    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / (log_name or "launcher.out.log")
    with open(log_path, "ab", buffering=0) as fh:
        fh.write(f"\n=== launch {time.strftime('%m-%d %H:%M:%S')} :: {' '.join(argv)} ===\n".encode("utf-8"))
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=FLAGS,
            close_fds=True,
        )
    print(f"launched pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
