# 星辰 · XingChen

从零训练一个 LLM 的最小完整框架，架构血统学自 DeepSeek 技术报告主线。
不依赖 transformers / sentencepiece / flash-attention，模型、分词器、训练器全部手写，
每一行都可以读懂、修改、做实验。

> 后续发展与能力提升方案见 [ROADMAP.md](ROADMAP.md)。

```text
数据 → BPE 分词 → 预训练（MLA + MoE + MTP + mHC）→ 检查点 → 缓存解码生成
```

## 与 DeepSeek 的血统对照

| 星辰模块 | 学自哪里 | 说明 |
|---|---|---|
| `MLAAttention` | DeepSeek-V2（arXiv:2405.04434） | 低秩压缩 KV：缓存只存 `c_kv + k_rope`，RoPE 解耦到独立维度 |
| `MoEFeedForward` 专家结构 | DeepSeekMoE（arXiv:2401.06066） | 细粒度小专家 + 始终激活的共享专家 |
| `e_bias` 负载均衡 | DeepSeek-V3（arXiv:2412.19437） | 无辅助损失均衡：sigmoid 打分、按偏置选人、按负载更新偏置、梯度不受污染 |
| `MTPModule` | DeepSeek-V3 | 多 token 预测：共享嵌入/输出头的浅层分支，预测 t+2 |
| `HyperConnection` | DeepSeek-V4 / mHC（arXiv:2512.24880） | n 条残差流 + 动态混合矩阵；流内矩阵经 Sinkhorn 迭代投影到双随机流形，保留恒等映射 |
| `scripts/prepare_data.py` 数据管线 | DeepSeek LLM（arXiv:2401.02954） | 整库编码成一维 token 流 + memmap 随机窗口采样 |
| 配置即插件（YAML 开关一切组件） | DeepSeek Harness（"Everything is a Plugin"） | 注意力/MoE/MTP/mHC 全部配置驱动，换组件不改代码 |

> DeepSeek 近期开源的 DeepInfer 是推理服务框架、DeepSeek Harness 是 Agent 运行时框架，
> 都不是训练框架；对"从零训练"最有价值的是上面的模型架构主线，星辰取的是这条线。

## 快速开始

```bash
# 用 ComfyUI 自带的 Python（已含 torch 2.14 + CUDA），也可用系统 Python 装好 torch
PY=/d/dsh-demo/ComfyUI_windows_portable/python_embeded/python.exe
cd /d/dsh-demo/xingchen

# 1. 训分词器 + 编码语料（换语料时改 --corpus 和 --vocab_size，需与 config 里 vocab_size 一致）
$PY scripts/prepare_data.py --corpus data/sample_corpus.txt --vocab_size 4096

# 2. 预训练 tiny（约 52M 参数，2.9GB 显存，与 ComfyUI 可共存）
$PY scripts/train.py --config configs/tiny.yaml                  # 完整 2000 步
$PY scripts/train.py --config configs/tiny.yaml --max_iters 60   # 快速冒烟

# 2.5 （可选）另开一个终端窗口，实时观看训练曲线（训练结束后也可回看）
$PY scripts/monitor.py        # 浏览器打开 http://127.0.0.1:8765，每 3 秒自动刷新

# 3. 续写 / 交互（--prompt 单次，省略则进 REPL）
$PY scripts/chat.py --ckpt checkpoints/latest.pt --prompt "夜空中的星星" --max_new 120

# 4. 断点续训
$PY scripts/train.py --config configs/tiny.yaml --resume checkpoints/latest.pt
```

**base 配置**（约 167M 参数，`configs/base.yaml`）：需 8GB 显存独占，训练前先关 ComfyUI。
真实语料请把 `--vocab_size` 提到 16k–32k，并同步改配置里的 `model.vocab_size`。

## 目录结构

```text
xingchen/
├── xingchen/
│   ├── config.py       # dataclass 配置 + YAML 加载（数值自动纠正）
│   ├── tokenizer.py    # 纯 Python 字节级 BPE（训练/编码/解码/存取）
│   ├── data.py         # uint32 bin + memmap 随机窗口采样
│   ├── model.py        # MLA / MoE / MTP / mHC / Block / 缓存
│   └── generation.py   # temperature / top-k / top-p / 重复惩罚 / n-gram 禁复现
├── scripts/
│   ├── prepare_data.py # 分词器训练 + 语料编码
│   ├── train.py        # 训练循环（AMP/梯度累积/余弦调度/续训/中途采样）
│   ├── chat.py         # 续写与 REPL
│   └── monitor.py      # 零依赖训练面板（浏览器看 loss 曲线）
├── configs/
│   ├── tiny.yaml       # 52M，全组件开启的演示配置
│   └── base.yaml       # 167M，按 V3/V4 比例缩放的"真训"配置
└── data/
    └── sample_corpus.txt  # 自带演示语料（约 2.6k 字）
```

## 实测记录（RTX 5060 Laptop 8GB）

- tiny 训练 20k token/s、峰值显存 2.9GB（batch 8 × seq 512）；500 步 train loss 8.59 → 0.005
- 自带语料只有 2150 token：train loss 会背到接近 0、验证 loss 回升是**必然**的（模型在背书）。
  想看"真学习"，换 ≥ 数十 MB 的语料；DeepSeek LLM 的预训练语料是 2 万亿 token 量级。
- 生成有碎片拼接感属正常：base 模型没做 SFT，且语料极小。

## 换真实语料

已内置中文维基百科全流程（实测 300MB 语料、143k 篇、92M token）：

```bash
# 1. 下载 parquet 分片（走 hf-mirror，选 2 个分片约 390MB）
curl -L -o data/wiki/raw/train-00002.parquet \
  "https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00002-of-00006.parquet"
curl -L -o data/wiki/raw/train-00003.parquet \
  "https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00003-of-00006.parquet"

# 2. 清洗成 txt（文章间用 =DOC= 分隔；4 秒）
$PY scripts/build_wiki_corpus.py --max_mb 300

# 3. 训 16k 分词器 + 并行编码（多进程，约 3 分钟）
$PY scripts/prepare_data.py --corpus data/wiki/corpus.txt --vocab_size 16384 --out_dir data/wiki --sample_mb 2

# 4. 训练（configs/tiny-wiki.yaml 已配好 24000 步 ≈ 98M token ≈ 90 分钟）
$PY scripts/train.py --config configs/tiny-wiki.yaml
```

其他中文开放语料（Skywork、OSCAR 等）同样走第 3 步之后的流程。
分词器为百 MB 级语料做过工程化：预切分（GPT-2 同思路）+ 堆增量训练 + 多进程编码。

**血泪教训**：手写注意力必须加因果掩码。漏掉时 loss 会"假性趋零"
（模型从未来位置抄答案，train/val 双双跌到 0.03），而生成看起来一切正常
（解码走 KV 缓存天然只见过去）——见到"好得不真实的 loss"先查这里。

## 扩容路线（对照 DeepSeek 开源栈）

| 方向 | 现状 | 下一步 |
|---|---|---|
| 注意力 | 手写朴素 MLA | FP8 训练与 FlashMLA 内核（DeepSeek open-infra） |
| 稀疏注意力 | 稠密 | DSA/CSA（V3.2/V4 的稀疏路线） |
| 专家并行 | 单卡 loop 分发 | DeepEP all-to-all、DualPipe 流水线 |
| 后训练 | 仅预训练 | SFT 模板 + GRPO（DeepSeekMath/R1 路线） |
| 推理解码 | 逐 token | 用已训练的 MTP 分支做投机解码 |
| 长上下文 | 固定 rope_base | YaRN 缩放 |

## 参考

- DeepSeek LLM: arXiv:2401.02954 · DeepSeekMoE: arXiv:2401.06066
- DeepSeek-V2 (MLA): arXiv:2405.04434 · DeepSeek-V3: arXiv:2412.19437
- mHC (V4 超连接): arXiv:2512.24880
- DeepSeek 开源基础设施: github.com/deepseek-ai（FlashMLA / DeepEP / DeepGEMM / 3FS）
