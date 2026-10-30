# -*- coding: utf-8 -*-
"""星辰 · GPU 温度看门狗：以电脑安全为主

逻辑（默认参数）：
  * 每 5 秒读取 GPU 温度
  * 温度 ≥ 80°C  → 暂停训练进程（suspend，GPU 空转散热）
  * 温度 ≤ 75°C  → 恢复训练（resume）
  * 温度 ≥ 90°C  → 强制终止训练进程（保卡优先）
  * 全程记录日志到 temp_watch.log

用法：
  python scripts/temp_watch.py                 # 自动发现占用显卡的 python 进程
  python scripts/temp_watch.py --pid 12345     # 指定训练进程
  python scripts/temp_watch.py --suspend_above 78 --kill_above 88
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = ROOT / "temp_watch.log"


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def nvidia_query(fields: str) -> str:
    # CREATE_NO_WINDOW：看门狗没有控制台时，子进程会弹黑窗；5 秒一次会一直闪
    _no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, creationflags=_no_window)
    return out.stdout.strip()


def find_training_pid() -> int | None:
    """找到占用显卡的 python 进程（即训练进程）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and "python" in parts[1].lower():
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--suspend_above", type=float, default=80.0)
    ap.add_argument("--resume_below", type=float, default=75.0)
    ap.add_argument("--kill_above", type=float, default=90.0)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    log(f"[watchdog] 启动：>={args.suspend_above}°C 暂停，<={args.resume_below}°C 恢复，"
        f">={args.kill_above}°C 终止训练；每 {args.interval}s 检查")

    pid = args.pid
    proc = None
    suspended = False
    while True:
        try:
            temp_s = nvidia_query("temperature.gpu,power.draw,utilization.gpu")
            parts = [x.strip() for x in temp_s.split(",")]
            temp = float(parts[0])
            power = parts[1] if len(parts) > 1 else "?"
            util = parts[2] if len(parts) > 2 else "?"
        except Exception as e:
            log(f"[watchdog] 读取温度失败：{e}")
            time.sleep(args.interval)
            continue

        # 确认目标进程
        if pid is None or proc is None or not psutil.pid_exists(pid):
            pid = find_training_pid()
            proc = psutil.Process(pid) if pid else None
            if proc:
                log(f"[watchdog] 盯上训练进程 PID={pid}")
            suspended = False

        state = "运行中"
        if proc is not None:
            try:
                if temp >= args.kill_above:
                    log(f"[watchdog] ⚠️ 温度 {temp}°C ≥ {args.kill_above}°C，强制终止训练进程！")
                    for c in proc.children(recursive=True):
                        c.kill()
                    proc.kill()
                    state = "已终止训练"
                    pid = None
                    proc = None
                    suspended = False
                elif temp >= args.suspend_above and not suspended:
                    proc.suspend()
                    suspended = True
                    log(f"[watchdog] 🔥 温度 {temp}°C ≥ {args.suspend_above}°C，暂停训练散热")
                    state = "已暂停(高温)"
                elif temp <= args.resume_below and suspended:
                    proc.resume()
                    suspended = False
                    log(f"[watchdog] ❄️ 温度 {temp}°C ≤ {args.resume_below}°C，恢复训练")
                    state = "已恢复"
                elif suspended:
                    state = "已暂停(等待降温)"
            except psutil.NoSuchProcess:
                log("[watchdog] 训练进程已退出（可能训练完成）")
                pid, proc, suspended = None, None, False
        else:
            state = "未发现训练进程"

        log(f"[watchdog] GPU {temp}°C | {power}W | 利用率 {util}% | {state}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
