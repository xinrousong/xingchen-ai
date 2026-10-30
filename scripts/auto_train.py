# -*- coding: utf-8 -*-
"""星辰 · 无人值守全自动训练流水线

串起 数据构建 → 编码入库 → 续训预训练 → SFT 四步，每步幂等（有产物即跳过），
任何一步失败立即中止并留下日志，可反复重跑（会从最近的检查点继续）。

用法（建议配合温度看门狗一起跑）：
  python scripts/auto_train.py                    # 全流程
  python scripts/auto_train.py --pretrain_iters 72000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = ROOT / "logs"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} [pipeline] {msg}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, cmd: list[str], log_name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(f"▶ {name} 开始：{' '.join(cmd[1:])}")
    t0 = time.time()
    with open(LOGS / log_name, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if p.returncode != 0:
        log(f"✖ {name} 失败（exit {p.returncode}），详见 logs/{log_name}，流水线中止")
        sys.exit(1)
    log(f"✔ {name} 完成，用时 {(time.time()-t0)/60:.1f} 分钟")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_iters", type=int, default=72000,
                    help="预训练目标总步数（从 24000 续训到该值）")
    ap.add_argument("--sft_iters", type=int, default=4500)
    ap.add_argument("--fresh", action="store_true",
                    help="清除完成标记完整重跑——更新了宪法/故事、想让她学习新记忆时用")
    args = ap.parse_args()

    if args.fresh:
        LOGS.mkdir(parents=True, exist_ok=True)
        for m in ("pretrain.done", "sft.done", "topup.done"):
            (LOGS / m).unlink(missing_ok=True)
        log("（--fresh）已清除完成标记：将完整重跑预训练与 SFT，让她学到最新记忆")

    corpus = ROOT / "data/wiki/corpus.txt"
    train_bin = ROOT / "data/wiki/train.bin"

    # 1. 语料构建（corpus.txt 已存在则跳过；重建需要 raw 分片——它们可能已被移到 U 盘）
    if corpus.exists() and corpus.stat().st_size > 1_000_000:
        log("⏭ 语料构建 跳过（corpus.txt 已在本地；原始分片在 U 盘，重造语料时需先取回）")
    else:
        run_step("语料构建", [PY, "-u", "scripts/build_wiki_corpus.py",
                            "--max_mb", "1000"], "corpus.log")

    # 2. 编码入库：train.bin 比 语料和核心记忆 都新则跳过
    core_dir = ROOT / "data/core"
    newest_core = 0.0
    if core_dir.exists():
        newest_core = max((p.stat().st_mtime for p in core_dir.rglob("*") if p.is_file()),
                          default=0.0)
    if train_bin.exists() and train_bin.stat().st_mtime > corpus.stat().st_mtime \
            and train_bin.stat().st_mtime > newest_core:
        log("⏭ 编码入库 跳过（语料与核心记忆都没变）")
    else:
        run_step("编码入库", [PY, "-u", "scripts/prepare_data.py",
                            "--corpus", "data/wiki/corpus.txt",
                            "--vocab_size", "16384", "--out_dir", "data/wiki",
                            "--init_tokenizer", "data/wiki/tokenizer.json",
                            "--core_dir", "data/core", "--core_repeat", "20",
                            "--workers", "8"], "prepare.log")

    # 2.5 SFT 数据（含核心问答；问答或指令数据有更新才重建）
    sft_tokens = ROOT / "data/sft/tokens.bin"
    qa_dir = ROOT / "data/core/qa"
    newest_qa = newest_core
    if sft_tokens.exists() and sft_tokens.stat().st_mtime > max(newest_qa, 0.0):
        log("⏭ SFT 数据 跳过（已是最新，含核心问答）")
    else:
        run_step("SFT 数据构建", [PY, "-u", "scripts/prepare_sft.py"], "prepare_sft.log")

    # 3. 续训预训练（幂等标记：logs/pretrain.done）
    if (LOGS / "pretrain.done").exists():
        log("⏭ 预训练 跳过（已有完成标记）")
    else:
        run_step("预训练续训", [PY, "-u", "scripts/train.py",
                              "--config", "configs/tiny-wiki.yaml",
                              "--resume", "checkpoints-wiki/latest.pt",
                              "--max_iters", str(args.pretrain_iters),
                              "--device", "cuda"], "pretrain.log")
        (LOGS / "pretrain.done").touch()

    # 4. SFT（用刚练好的预训练模型）
    if (LOGS / "sft.done").exists():
        log("⏭ SFT 跳过（已有完成标记）")
    else:
        run_step("SFT 微调", [PY, "-u", "scripts/train_sft.py",
                            "--init", "checkpoints-wiki/latest.pt",
                            "--max_iters", str(args.sft_iters),
                            "--out_dir", "checkpoints-sft-v2",
                            "--device", "cuda"], "sft.log")
        (LOGS / "sft.done").touch()

    # 5. 记忆巩固：SFT 的大量新数据会冲掉背诵（灾难性遗忘），
    #    最后一步在"课文+问答混合"的密集集上短训，记忆和应题两头都保住
    topup_bin = ROOT / "data/wiki/core_topup.bin"
    if topup_bin.exists() and not (LOGS / "topup.done").exists():
        run_step("记忆巩固", [PY, "-u", "scripts/train.py",
                            "--config", "configs/tiny-wiki.yaml",
                            "--resume", "checkpoints-sft-v2/latest.pt",
                            "--out_dir", "checkpoints-sft-v2",
                            "--train_bin", "data/wiki/core_topup.bin",
                            "--lr", "3.0e-4",
                            "--max_iters", str(args.sft_iters + 2400),
                            "--device", "cuda"], "topup.log")
        (LOGS / "topup.done").touch()
    else:
        log("⏭ 记忆巩固 跳过（无巩固集或已完成）")

    # 6. 认知自检：考固定题目，验证宪法与故事是否记住
    run_step("认知自检", [PY, "-u", "scripts/core_check.py",
                        "--ckpt", "checkpoints-sft-v2/latest.pt"], "core_check_run.log")

    log("🎉 全流水线完成：checkpoints-sft-v2/latest.pt 是最终对话模型（已通过认知自检环节）")


if __name__ == "__main__":
    main()
