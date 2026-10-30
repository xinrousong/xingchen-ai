# 星辰 · 训练档案（MANIFEST）

> 生成于 2026-10-30。记录**不进 git 的大文件**（权重 / 语料）的位置、大小与 SHA256，
> 方便以后校验完整性、确认“某个模型是哪一次训练出来的”。

## 一、模型权重（不在仓库里，用哈希校验）

| 文件 | 大小 | SHA256 | 说明 |
|---|---|---|---|
| `checkpoints-base-wiki/latest.pt` | 1,997,199,103 B（1.86 GB） | `24525a64b9d326c88158878b407e6a67c3893b2718c402bbb28edc8e4cd921d8` | **base 重修终版**：iter 36000，验证 loss 3.06 |
| `E:\xingchen\checkpoints-base-final\latest-36000.pt` | 同上 | 同上 | 上面那个的备份（同字节） |
| `checkpoints-sft-v2/latest.pt` | 684,864,655 B（653 MB） | `5d3d71c52d3eef60cda9d804a34c118b8a78fa7aad91687457cdb0e9a790de2d` | tiny 对话版（SFT + 记忆巩固到 12400 步） |
| `checkpoints-wiki/latest.pt` | 684,864,655 B（653 MB） | `5266e2f8b2c248665de79c63d7b41737082671ba89343a77c11e31f789988ffa` | tiny 预训练底模（72000 步） |

**校验方法**（PowerShell）：

```powershell
(Get-FileHash 'checkpoints-base-wiki\latest.pt' -Algorithm SHA256).Hash.ToLower()
```

## 二、语料与 token 流（E 盘，可重建）

| 文件 | 大小 | SHA256 | 说明 |
|---|---|---|---|
| `E:\xingchen\data\wiki\train.bin` | 1,219,357,056 B（1.16 GB） | `11e28641ddb68cedc636102fcff18a2f33bbbd6998c26586d42a97783cf38d4e` | 3.05 亿 token 训练流（含核心记忆注入） |
| `E:\xingchen\data\wiki\corpus.txt` | 1,016,869,001 B（970 MB） | `756790035d95d8fec3c2a132c27b4b3caaeb8d370bd681190d86297ea1d9a6df` | 原始中文维基文本（`=DOC=` 分隔） |
| `E:\xingchen\data\wiki\raw\*.parquet` | 1.08 GB（5 片） | — | 维基原始分片，`corpus.txt` 由它生成 |

重建方式：`python scripts/build_wiki_corpus.py --max_mb 1000` → `python scripts/prepare_data.py --corpus data/wiki/corpus.txt --vocab_size 16384 --out_dir data/wiki --init_tokenizer data/wiki/tokenizer.json --core_dir data/core --core_repeat 20 --workers 8`

## 三、这次 base 重修的关键数字

| 项 | 值 |
|---|---|
| 配置 | `configs/base-wiki.yaml`，166.37M 参数，seq 512，batch 2，grad_accum 8（8192 token/步） |
| 训练区间 | iter 10001 → 36000（从 8000 步存档续训） |
| 训练 loss | 3.565（10k–12k 均值）→ **2.907**（34k–36k 均值），末条 2.73 |
| 验证 loss | 4.05（起步）→ **3.06**（36000 步，26 次 eval） |
| 吞吐 | **约 4500 tok/s**（向量化分派生效后；改之前是 500–1000） |
| 耗时 | 约 13.1 小时（10-30 00:0x → 13:13） |
| 精读日志 | `checkpoints-base-wiki/train_log.jsonl`（2601 条 train + 26 条 eval，每 10 步一条） |

## 四、为什么这些大文件不进 git

GitHub 单文件硬限 **100 MB**，仓库建议 < 1 GB；Git LFS 免费额度只有 1 GB 存储 + 1 GB/月流量。
一个 1.86 GB 的检查点就会爆掉，而且 clone 会极慢。所以：

- **仓库里**：代码（约 2300 行）、配置、文档、`train_log.jsonl` 曲线、日志文本
- **仓库外**：权重（本机 + E 盘备份）、语料分片（E 盘）

需要复现某次实验时，用本文件的哈希确认权重版本，再按 `ROADMAP.md` / `后续执行手册.md` 的命令重跑。
