# 星辰 · 架构说明（实测版）

> 数据来源：`checkpoints-base-wiki/latest.pt`（iter 35999）逐键统计，不是配置文件推算。
> 对应代码：`xingchen/model.py`（556 行）。设计取向是"每一行都能读懂"，所以刻意不做 kernel 级优化。

## 一、一句话规格

**166.37M 参数的 MoE 解码器**：10 层、dim 768、12 头、词表 16384、tied embedding，
MLA + MoE + MTP + mHC 四件套全开。

## 二、参数分布（实测）

| 模块 | 参数量 | 占比 |
|---|---|---|
| MoE 路由专家 ×12（9 层） | 95.55M | 57.4% |
| 词嵌入（与输出头 tied） | 12.58M | 7.6% |
| MLA 注意力（10 层） | 21.88M | 13.2% |
| MoE 共享专家 ×2（9 层） | 15.93M | 9.6% |
| MTP 分支 | 15.77M | 9.5% |
| 第 0 层 dense FFN | 4.42M | 2.7% |
| mHC 超连接（10 层合计） | 0.14M | 0.09% |
| 路由 gate | 0.08M | 0.05% |
| **合计** | **166.37M** | |

> 注：`state_dict` 逐键相加会得 178.95M，多出的 12.58M 是 `tie_embeddings` 让
> `lm_head.weight` 与 `embed.weight` 指向同一张量、被数了两遍。

## 三、前向路径

```text
idx → Embed(16384×768)
     → streams[n_streams=2, 768]      主流载入嵌入，其余流从零开始
     → Block ×10
          mHC 外壳：norm(2×768) → φ_pre/φ_post/φ_res → H_pre / H_post / H_res
                    x_in = H_pre · streams
                    ├─ RMSNorm → MLA      → +残差
                    └─ RMSNorm → FFN(MoE) → +残差
                    new  = H_res · streams + H_post ⊗ y
     → 取主流 streams[0] → final_norm → lm_head(tied)
```

第 0 层用 dense FFN（`first_k_dense: 1`），第 1~9 层用 MoE。

## 四、四个模块的实现要点

### MLA（多头潜在注意力，学自 DeepSeek-V2）

- 低秩压缩：`q: 768→512→12×(64+32)`，`kv: 768→(256+32)`，再由 `kv_b` 升维出 k_nope 与 v
- **RoPE 只作用在 32 维解耦部分**，nope 部分不带位置信息
- 训练走 SDPA 快速路径（把 `1/√d` 折进 q，`scale=1.0`）；推理走缓存路径
- **KV 缓存每 token 每层只存 `c_kv(256) + k_rope(32) = 288` 个数**
  同等 MHA 要存 `12×(64+32+64) = 1920` 个 → **省 6.7 倍**

### MoE（学自 DeepSeekMoE / V3）

- 结构：12 个路由专家 + 2 个始终激活的共享专家，`top_k=4`，`expert_dim=384`
- 路由：sigmoid 打分，用 `score + e_bias` 选人，**门控权重取原始 score 再归一化**
  → `e_bias` 只影响路由、不污染梯度（V3 的 noaux_tc 思路）
- **无辅助损失均衡**：`e_bias` 由训练器按专家负载符号规则更新
- 分发：`_dispatch_grouped` 按专家排序 → 一次 `bincount` 拿偏移 → 连续切片喂专家 → `index_add` 回填。
  **全层只有 1 次 GPU→CPU 同步**，实测比逐专家 loop 快一倍（8.10 → 3.79 ms/层）

### mHC 流形约束超连接（学自 DeepSeek-V4）

- 残差流扩为 `n_streams=2`；每层按当前流状态动态生成三个连接矩阵：
  `H_pre(1×n)` 读出、`H_post(1×n)` 写入、`H_res(n×n)` 流内混合
- `H_res` 经 **Sinkhorn 迭代投影到双随机矩阵流形**（当前配置 6 次，原 20 次）
- 初始化：φ 权重全零 + 偏置 one-hot + `α=0.01` → **训练初期严格等价于标准残差**，随训练逐步学会跨流读写
- 代码量很小（0.14M 参数）但 kernel 数多：每层 `2×iters` 次归一化，是隐性开销

### MTP 多 token 预测（学自 DeepSeek-V3）

- 深度 1 的浅层分支：`proj(concat(norm(h_t), norm(emb_{t+1})))` → 一个 MoE Block → 预测 t+2
- 共享词嵌入与输出头，`loss_weight 0.3` 作为辅助损失
- **投机解码的草稿模型尚未接入**（ROADMAP M2），所以当前只贡献训练信号

## 五、三个必须知道的后果

| 现象 | 原因 | 影响 |
|---|---|---|
| **激活率仅 43%** | 12 选 4 + 2 共享 | 总参 166M，前向实际参与约 **87M**（含 MTP），纯推理约 72M。它不是"166M 的模型"，更接近"70M 级模型 + 一堆专家容量" |
| **KV 缓存省 6.7 倍** | MLA 的低秩 c_kv + 解耦 RoPE | 长上下文推理的成本大幅下降，这是 MLA 的主要动机 |
| **`max_seq_len=4096` 但只训了 512** | `rope_base` 固定 10000，未上 YaRN | 512 以上的位置编码属于"没学过"的状态，直接拉长会掉质量 |

## 六、当前配置（`configs/base-wiki.yaml`）

| 项 | 值 |
|---|---|
| 序列 / batch / 累积 | 512 × 2 × 8 = **8192 token/步** |
| 总步数 / 总 token | 36000 步 ≈ **2.95 亿 token** ≈ 1.8 token/参数 |
| 学习率 | 3e-4 → 3e-5（cosine，warmup 500） |
| 精度 / 检查点 | bf16 autocast；梯度检查点关闭（seq 512 够用） |
| 实测吞吐 | **4.5k tok/s**（RTX 5060 Laptop 8GB，向量化分发后） |
| 验证 loss | 4.05 → **3.0617**（26 次 eval） |

> 下一轮（1B 语料）用 `configs/base-wiki-1b.yaml`：122000 步 ≈ 10 亿 token，
> `sinkhorn_iters` 降到 6，`grad_checkpoint` + `ce_chunk` 全开。

## 七、已知短板

1. **序列只有 512**：中文维基单篇中位 586 汉字，大量文档被拦腰截断
2. **分词器 16k 偏低效**：约 0.9 token/字，升到 32k 可省 ~25% token
3. **注意力是稠密全量**：无 FlashMLA / 无稀疏注意力，长上下文不可用
4. **单卡专家分发**：MoE 只做了卡内向量化，没有专家并行
5. **训练量不足**：1.8 token/参数，距 Chinchilla 的 20 差一个数量级

后续修补顺序见 `ROADMAP.md`，性能优化记录见 `优化记录-20261116.md`。
