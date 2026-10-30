# -*- coding: utf-8 -*-
"""星辰 · 从零训练 LLM 框架（DeepSeek 血统）"""
from .config import (DataConfig, HyperConnectionConfig, MLAConfig, ModelConfig,
                     MoEConfig, MTPConfig, TrainConfig, XingChenConfig, load_config)
from .tokenizer import ByteBPETokenizer
from .data import binarize, get_batch, load_bin
from .model import XingChen, XingChenCache, param_summary
from .generation import generate, generate_speculative

__version__ = "0.1.0"

__all__ = [
    "XingChen", "XingChenCache", "param_summary", "generate", "generate_speculative",
    "XingChenConfig", "load_config", "ByteBPETokenizer",
    "binarize", "get_batch", "load_bin",
    "MLAConfig", "MoEConfig", "MTPConfig", "HyperConnectionConfig",
    "ModelConfig", "TrainConfig", "DataConfig",
]
