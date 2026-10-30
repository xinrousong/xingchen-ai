# -*- coding: utf-8 -*-
"""星辰 · 配置定义

所有组件（注意力 / MoE / MTP / mHC）均由配置开关，借鉴 DeepSeek Harness
"一切皆插件"的思想：换组件不改代码，只改 YAML。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MLAConfig:
    """多头潜在注意力（Multi-head Latent Attention），出自 DeepSeek-V2。

    KV 缓存只保留低秩压缩向量 c_kv 与解耦 RoPE 键 k_rope，
    每 token 缓存大小 = kv_lora_rank + qk_rope_head_dim。
    """
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32


@dataclass
class MoEConfig:
    """细粒度专家 + 共享专家（DeepSeekMoE），无辅助损失负载均衡（V3）。

    e_bias 是不参与梯度的路由偏置缓冲，由训练器按负载符号规则更新，
    实现 V3 的 auxiliary-loss-free balancing。
    """
    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    expert_dim: int = 768
    norm_topk_prob: bool = True
    balance_gamma: float = 1e-3      # V3 论文取值
    balance_window: int = 1          # 每多少个优化器步更新一次偏置


@dataclass
class MTPConfig:
    """多 token 预测（Multi-Token Prediction），出自 DeepSeek-V3。

    额外一个浅层模块，与主模型共享词嵌入与输出头，预测 t+2 处的 token；
    训练后可关掉，仅作深监督用。
    """
    enabled: bool = True
    loss_weight: float = 0.3


@dataclass
class HyperConnectionConfig:
    """流形约束超连接（mHC），出自 DeepSeek-V4（arXiv:2512.24880）。

    把单条残差流扩为 n_streams 条，层间用动态混合矩阵读写；
    流内混合矩阵经 Sinkhorn 迭代投影到双随机矩阵流形，保留恒等映射性质。
    """
    enabled: bool = True
    n_streams: int = 2               # V4 论文用 4；tiny 演示取 2
    sinkhorn_iters: int = 20         # 论文实测 20 次近似收敛
    alpha_init: float = 0.01         # 论文附录 A.1


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn: str = "moe"                 # "moe" | "dense"
    first_k_dense: int = 1           # 前 k 层用稠密 FFN（V3 前 3 层为稠密）
    dense_intermediate: int = 0      # 0 = 自动取 round(2.5*dim / 64)*64
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    max_seq_len: int = 4096          # 由 load_config 依据训练 seq_len 自动放宽
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    hc: HyperConnectionConfig = field(default_factory=HyperConnectionConfig)


@dataclass
class TrainConfig:
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    max_iters: int = 2000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 20
    ckpt_interval: int = 500
    sample_interval: int = 200
    log_interval: int = 10
    amp: str = "bf16"                # "bf16" | "off"
    seed: int = 2026
    out_dir: str = "checkpoints"


@dataclass
class DataConfig:
    tokenizer_path: str = "data/tokenizer.json"
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"


@dataclass
class XingChenConfig:
    name: str = "xingchen-tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_dataclass(dc_cls, data: Optional[dict]):
    """把 dict（含嵌套）合并进 dataclass 默认值，未给出的字段保持默认。"""
    import dataclasses
    if data is None:
        return dc_cls()
    fields = {f.name: f for f in dataclasses.fields(dc_cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in fields:
            raise KeyError(f"未知配置项: {dc_cls.__name__}.{key}")
        default = getattr(dc_cls(), key)
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _merge_dataclass(type(default), value)
        elif isinstance(default, (int, float)) and isinstance(value, str):
            # YAML 1.1 会把 6e-4 这类无小数点的科学计数法解析成字符串
            kwargs[key] = type(default)(float(value)) if isinstance(default, float) else int(float(value))
        else:
            kwargs[key] = value
    return dc_cls(**kwargs)


def load_config(path: str | Path) -> XingChenConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = _merge_dataclass(XingChenConfig, raw)
    # RoPE 表覆盖到 训练窗口 + 生成余量
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, cfg.train.seq_len + 2048)
    if cfg.model.dense_intermediate <= 0:
        cfg.model.dense_intermediate = int((cfg.model.dim * 2.5) // 64 * 64)
    if cfg.model.ffn == "moe":
        assert cfg.model.moe.top_k <= cfg.model.moe.n_routed
    return cfg


def with_mhc_disabled(model_cfg: ModelConfig) -> ModelConfig:
    """派生一份关闭 mHC 的模型配置（供 MTP 浅层模块等使用）。"""
    return replace(model_cfg, hc=replace(model_cfg.hc, enabled=False))
