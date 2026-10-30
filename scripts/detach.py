# -*- coding: utf-8 -*-
"""把长跑任务从当前终端 / 编辑器 / 任何父进程里彻底摘出去。

为什么需要它：直接 start / Start-Process 出来的子进程会跟父进程共用控制台，
父进程退出或被 Ctrl+C 时，子进程会收到 CTRL_C_EVENT（退出码 0xC000013A），
所以“关掉编辑器 / 关掉助手 / 关掉命令行窗口，训练就跟着死”。
这里用 DETACHED_PROCESS（不带控制台）+ CREATE_NEW_PROCESS_GROUP，
并尝试 CREATE_BREAKAWAY_FROM_JOB 跳出父进程的作业对象。

用法（在 xingchen 目录下）：
    python scripts/detach.py train
    python scripts/detach.py watch
    python scripts/detach.py monitor
    python scripts/detach.py all
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT.parent / "ComfyUI_windows_portable" / "python_embeded" / "python.exe"
LOGS = ROOT / "logs"

JOBS = {
    "train": (
        ["-u", "scripts/train.py", "--config", "configs/base-wiki.yaml",
         "--resume", "checkpoints-base-wiki/latest.pt", "--device", "cuda"],
        "base_m1.log",
    ),
    "watch": (["-u", "scripts/temp_watch.py"], "watchdog.out.log"),
    "monitor": (
        ["-u", "scripts/monitor.py", "--log", "checkpoints-base-wiki/train_log.jsonl"],
        "monitor.out.log",
    ),
}

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

FLAG_SETS = (
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
)


def launch(name: str) -> int:
    args, logname = JOBS[name]
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / logname
    with open(log_path, "ab", buffering=0) as fh:
        fh.write(f"\n=== detach {name} {time.strftime('%m-%d %H:%M:%S')} ===\n".encode("utf-8"))
        for flags in FLAG_SETS:
            try:
                proc = subprocess.Popen(
                    [str(PY), *args],
                    cwd=str(ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                    close_fds=True,
                )
            except OSError as exc:  # 作业对象不允许 breakaway 时会 PermissionError
                print(f"{name}: flags=0x{flags:08X} 失败：{exc}", file=sys.stderr)
                continue
            print(f"{name}: pid={proc.pid} flags=0x{flags:08X} log={log_path.name}")
            return 0
    return 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in (*JOBS, "all"):
        print(__doc__)
        return 2
    names = list(JOBS) if sys.argv[1] == "all" else [sys.argv[1]]
    rc = 0
    for name in names:
        rc |= launch(name)
        time.sleep(3 if name == "train" else 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
