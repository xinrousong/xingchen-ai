# -*- coding: utf-8 -*-
"""星辰 · 模型定义（DeepSeek 血统，从零实现）

架构成分与出处：
  * MLA  多头潜在注意力          —— DeepSeek-V2 (arXiv:2405.04434)
  * MoE  细粒度专家 + 共享专家    —— DeepSeekMoE (arXiv:2401.06066)
  *      无辅助损失负载均衡        —— DeepSeek-V3 (arXiv:2412.19437)
  * MTP  多 token 预测           —— DeepSeek-V3
  * mHC  流形约束超连接           —— DeepSeek-V4 (arXiv:2512.24880)

所有实现刻意保持朴素直白（手写注意力、显式专家分发），便于逐行读懂；
生产级替代（FlashMLA / DeepEP 等）见 README 扩容路线。
"""
from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 快速路径开关（默认开）。环境变量可逐项回退，便于 A/B 与回归对照：
#   XINGCHEN_MOE_DISPATCH=loop    → 用原来的逐专家 Python 分发
#   XINGCHEN_MLA_ATTN=naive       → 用原来的手写注意力（不调 SDPA）
_MOE_DISPATCH = os.environ.get("XINGCHEN_MOE_DISPATCH", "vectorized").strip().lower()
_MLA_ATTN = os.environ.get("XINGCHEN_MLA_ATTN", "sdpa").strip().lower()

from .config import MLAConfig, ModelConfig, with_mhc_disabled


# --------------------------------------------------------------------------- 基础件
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """解耦 RoPE：只作用于 MLA 的 rope 维度。"""

    def __init__(self, head_dim: int, base: float, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor):
        return self.cos[positions], self.sin[positions]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)，cos/sin: (T, D/2)。半分旋转。"""
    x1, x2 = x.chunk(2, dim=-1)
    c = cos.to(x.dtype)[None, None]
    s = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Expert(nn.Module):
    """SwiGLU 专家。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseFeedForward(nn.Module):
    """稠密 SwiGLU 前馈（V3 前若干层与 dense 模式使用）。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.expert = Expert(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class MoEFeedForward(nn.Module):
    """DeepSeekMoE：细粒度路由专家 + 始终激活的共享专家 + 无辅助损失均衡。

    路由打分用 sigmoid（V3 的 noaux_tc 风格）：选择按 score + e_bias，
    门控权重用原始 score 归一化，因此 e_bias 只影响路由、不污染梯度；
    e_bias 由训练器按专家负载符号规则更新（update_expert_bias）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        m = cfg.moe
        self.n_routed = m.n_routed
        self.top_k = m.top_k
        self.norm_topk_prob = m.norm_topk_prob
        self.gate = nn.Linear(cfg.dim, m.n_routed, bias=False)
        self.experts = nn.ModuleList(Expert(cfg.dim, m.expert_dim) for _ in range(m.n_routed))
        self.shared = nn.ModuleList(Expert(cfg.dim, m.expert_dim) for _ in range(max(1, m.n_shared)))
        self.register_buffer("e_bias", torch.zeros(m.n_routed), persistent=True)
        self.register_buffer("expert_counts", torch.zeros(m.n_routed, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        xf = x.view(-1, C)                                   # (N, C)
        scores = torch.sigmoid(self.gate(xf).float())        # (N, E)
        top_w, top_i = torch.topk(scores + self.e_bias.float(), self.top_k, dim=-1)
        top_w = scores.gather(-1, top_i)                     # 权重取原始得分
        if self.norm_topk_prob and self.top_k > 1:
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)

        y = xf
        for shared in self.shared:
            y = y + shared(xf)

        if _MOE_DISPATCH == "loop":
            routed = self._dispatch_loop(xf, top_w, top_i)
        else:
            routed = self._dispatch_grouped(xf, top_w, top_i)
        return (y + routed).view(B, T, C)

    def _dispatch_loop(self, xf: torch.Tensor, top_w: torch.Tensor, top_i: torch.Tensor) -> torch.Tensor:
        """朴素实现：逐专家布尔掩码 + nonzero。每层 12 次 GPU→CPU 同步，慢但直白。"""
        routed = torch.zeros_like(xf)
        counting = self.training and torch.is_grad_enabled()
        for e in range(self.n_routed):
            mask = top_i == e                                # (N, k)
            if not mask.any():
                continue
            token_idx = mask.any(-1).nonzero(as_tuple=True)[0]
            w = (top_w * mask).sum(-1)[token_idx].to(routed.dtype)
            routed.index_add_(0, token_idx, self.experts[e](xf[token_idx]) * w.unsqueeze(-1))
            if counting:
                with torch.no_grad():
                    self.expert_counts[e] += token_idx.numel()
        return routed

    def _dispatch_grouped(self, xf: torch.Tensor, top_w: torch.Tensor, top_i: torch.Tensor) -> torch.Tensor:
        """按专家排序后连续切片计算：整个分派只做 1 次同步，算子数与专家数无关。

        与 _dispatch_loop 数学等价（同一组 (token, expert, 权重) 三元组，只是求和顺序不同）。
        """
        N, K = top_i.shape
        flat_e = top_i.reshape(-1)                                  # (N*k,)
        flat_t = torch.arange(N, device=xf.device).repeat_interleave(K)
        flat_w = top_w.reshape(-1).to(xf.dtype)

        order = torch.argsort(flat_e, stable=True)
        e_sorted = flat_e[order]
        t_sorted = flat_t[order]
        w_sorted = flat_w[order]

        counts = torch.bincount(e_sorted, minlength=self.n_routed)  # (E,)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.expert_counts += counts

        offsets = counts.cumsum(0).tolist()                         # 全层仅此一次同步
        x_sorted = xf.index_select(0, t_sorted)                     # (N*k, C)

        outs = []
        start = 0
        for e in range(self.n_routed):
            end = int(offsets[e])
            if end > start:
                outs.append(self.experts[e](x_sorted[start:end]))
            start = end
        if not outs:
            return torch.zeros_like(xf)
        out = torch.cat(outs) * w_sorted.unsqueeze(-1)
        return torch.zeros_like(xf).index_add_(0, t_sorted, out)


# --------------------------------------------------------------------------- MLA 注意力
class MLAAttention(nn.Module):
    """多头潜在注意力。

    训练：c_kv = kv_a_norm(kv_a(x)) 低秩压缩；k_nope / v 由 c_kv 经 kv_b 升维得到，
    RoPE 仅作用于解耦的 q_rope / k_rope 维度（V2 §2.1）。
    推理：KV 缓存只存 (c_kv, k_rope)，每 token 共 kv_lora_rank + qk_rope_head_dim 个量。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        a = cfg.mla
        C, H = cfg.dim, cfg.n_heads
        self.H, self.dn, self.dr, self.dv = H, a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim
        self.q_a = nn.Linear(C, a.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(a.q_lora_rank, cfg.norm_eps)
        self.q_b = nn.Linear(a.q_lora_rank, H * (self.dn + self.dr), bias=False)
        self.kv_a = nn.Linear(C, a.kv_lora_rank + self.dr, bias=False)
        self.kv_a_norm = RMSNorm(a.kv_lora_rank, cfg.norm_eps)
        self.kv_b = nn.Linear(a.kv_lora_rank, H * (self.dn + self.dv), bias=False)
        self.o_proj = nn.Linear(H * self.dv, C, bias=False)

    def forward(self, x, cos, sin, layer_cache=None, offset: int = 0):
        B, T, C = x.shape
        H, dn, dr, dv = self.H, self.dn, self.dr, self.dv

        q = self.q_b(self.q_a_norm(self.q_a(x)))
        q = q.view(B, T, H, dn + dr).permute(0, 2, 1, 3)     # (B,H,T,dn+dr)
        q_nope, q_rope = q.split([dn, dr], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        kv = self.kv_a(x)                                    # (B,T,kv_lora+dr)
        c_kv, k_rope = kv.split([self.kv_a.out_features - dr, dr], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)   # (B,1,T,dr)

        if layer_cache is not None:
            layer_cache.write(offset, c_kv, k_rope.squeeze(1))
            c_all = layer_cache.ckv[:, :offset + T]          # (B,S,kv_lora)
            kr_all = layer_cache.krope[:, :offset + T].unsqueeze(1)  # (B,1,S,dr)
        else:
            c_all, kr_all = c_kv, k_rope                     # k_rope 保持 (B,1,T,dr)

        kv_lora = c_all.shape[-1]
        w_kc = self.kv_b.weight[: H * dn].view(H, dn, kv_lora)
        w_uv = self.kv_b.weight[H * dn:].view(H, dv, kv_lora)
        k_nope = torch.einsum("btk,hgk->bhtg", c_all, w_kc)  # (B,H,S,dn)
        v = torch.einsum("btk,hgk->bhtg", c_all, w_uv)       # (B,H,S,dv)

        # 快速路径：整段前向（训练时就是这条）交给 SDPA。
        # 两段的 1/sqrt(d) 缩放折进 q，等价于原来的 "q_nope@k_nope/sqrt(dn) + q_rope@k_rope/sqrt(dr)"。
        if _MLA_ATTN == "sdpa" and layer_cache is None and offset == 0 \
                and T == k_nope.shape[-2]:
            q_cat = torch.cat([q_nope * (dn ** -0.5), q_rope * (dr ** -0.5)], dim=-1)
            k_cat = torch.cat([k_nope, kr_all.expand(-1, H, -1, -1)], dim=-1)
            out = F.scaled_dot_product_attention(q_cat, k_cat, v, is_causal=True, scale=1.0)
            out = out.transpose(1, 2).reshape(B, T, H * dv)
            return self.o_proj(out)

        attn = (q_nope @ k_nope.transpose(-2, -1)) * (dn ** -0.5)
        attn = attn + (q_rope @ kr_all.transpose(-2, -1)) * (dr ** -0.5)
        # 因果掩码：第 i 个 query（绝对位置 offset+i）只能看见 ≤ 自身的 key。
        # 没有它，训练时模型能从未来位置"抄答案"，loss 假性趋零。
        q_pos = torch.arange(offset, offset + T, device=x.device)
        k_pos = torch.arange(k_nope.shape[-2], device=x.device)
        allow = k_pos[None, :] <= q_pos[:, None]                 # (T, S) bool
        attn = attn.masked_fill(~allow, float("-inf"))
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = attn @ v                                       # (B,H,T,dv)
        out = out.transpose(1, 2).reshape(B, T, H * dv)
        return self.o_proj(out)


# --------------------------------------------------------------------------- mHC
class HyperConnection(nn.Module):
    """流形约束超连接（mHC, arXiv:2512.24880）。

    残差流扩为 n 条；每层依据当前流状态动态生成三个连接矩阵：
      H_pre  (1×n)  读出：把 n 条流加权聚合成层输入
      H_post (1×n)  写入：把层输出写回各条流（2·sigmoid 保证幅度）
      H_res  (n×n)  流内混合：经 Sinkhorn 迭代投影到双随机矩阵流形
    初始化为恒等映射（φ 零初始化、偏置 one-hot、α=0.01），训练早期等价于
    标准残差，随训练逐渐学会跨流读写。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hc = cfg.hc
        self.n = hc.n_streams
        self.iters = hc.sinkhorn_iters
        nC = self.n * cfg.dim
        self.norm = RMSNorm(nC, cfg.norm_eps)
        self.phi_pre = nn.Linear(nC, self.n, bias=False)
        self.phi_post = nn.Linear(nC, self.n, bias=False)
        self.phi_res = nn.Linear(nC, self.n * self.n, bias=False)
        for p in (self.phi_pre, self.phi_post, self.phi_res):
            nn.init.zeros_(p.weight)
        self.alpha_pre = nn.Parameter(torch.tensor(float(hc.alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(hc.alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(hc.alpha_init)))

        def onehot_logits(size, hot_idx):
            v = torch.full((size,), -6.0)
            v[hot_idx] = 6.0
            return v

        b_pre = onehot_logits(self.n, 0)
        b_post = onehot_logits(self.n, 0)
        b_res = onehot_logits(self.n * self.n, 0 * self.n + 0)   # 对角元 (0,0)
        eye = torch.eye(self.n).reshape(-1) == 1
        b_res[eye] = 6.0
        self.register_buffer("b_pre", b_pre, persistent=True)
        self.register_buffer("b_post", b_post, persistent=True)
        self.register_buffer("b_res", b_res, persistent=True)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        M = torch.exp(logits.float())
        for _ in range(self.iters):
            M = M / M.sum(-1, keepdim=True).clamp_min(1e-12)     # 行归一
            M = M / M.sum(-2, keepdim=True).clamp_min(1e-12)     # 列归一
        return M

    def forward(self, streams: torch.Tensor, sublayer) -> torch.Tensor:
        B, T, n, C = streams.shape
        xh = self.norm(streams.reshape(B, T, n * C))             # (B,T,nC)
        pre = torch.sigmoid(self.alpha_pre * self.phi_pre(xh) + self.b_pre)
        post = 2.0 * torch.sigmoid(self.alpha_post * self.phi_post(xh) + self.b_post)
        res_logits = (self.alpha_res * self.phi_res(xh) + self.b_res).view(B, T, n, n)
        H_res = self._sinkhorn(res_logits).to(streams.dtype)

        x_in = torch.einsum("btj,btjc->btc", pre.to(streams.dtype), streams)
        y = sublayer(x_in)
        new = torch.einsum("btmj,btjc->btmc", H_res, streams)
        new = new + post.to(streams.dtype).unsqueeze(-1) * y.unsqueeze(2)
        return new


# --------------------------------------------------------------------------- Block / MTP
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = MLAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.ffn == "dense" or layer_idx < cfg.first_k_dense:
            self.ffn = DenseFeedForward(cfg.dim, cfg.dense_intermediate)
        else:
            self.ffn = MoEFeedForward(cfg)
        self.hc = HyperConnection(cfg) if cfg.hc.enabled else None

    def _sublayer(self, x, cos, sin, layer_cache, offset):
        x = x + self.attn(self.attn_norm(x), cos, sin, layer_cache, offset)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(self, x, cos, sin, layer_cache=None, offset: int = 0):
        if self.hc is not None:
            # x 为 (B,T,n,C) 的残差流张量
            return self.hc(x, lambda xin: self._sublayer(xin, cos, sin, layer_cache, offset))
        return self._sublayer(x, cos, sin, layer_cache, offset)


class MTPModule(nn.Module):
    """V3 的 MTP 浅层模块（深度 1）：共享词嵌入与输出头，预测 t+2。

    输入 = 投影( concat( norm(h_t), norm(emb_{t+1}) ) )，过一个普通 Block；
    生成阶段可用作投机解码的草稿模型（本仓库暂未实现，见 README 路线）。
    """

    def __init__(self, cfg: ModelConfig, rotary: RotaryEmbedding):
        super().__init__()
        C = cfg.dim
        self.emb_norm = RMSNorm(C, cfg.norm_eps)
        self.in_norm = RMSNorm(C, cfg.norm_eps)
        self.proj = nn.Linear(2 * C, C, bias=False)
        self.out_norm = RMSNorm(C, cfg.norm_eps)
        self.block = Block(with_mhc_disabled(cfg), layer_idx=cfg.first_k_dense)
        self.rotary = rotary
        self._infer_cache = None        # 推理用 MTP 缓存（懒创建）

    def reset_infer_cache(self) -> None:
        self._infer_cache = None

    def forward(self, h_pre: torch.Tensor, next_emb: torch.Tensor,
                use_cache: bool = False) -> torch.Tensor:
        B, T, C = h_pre.shape
        e = self.emb_norm(next_emb)
        z = self.proj(torch.cat([self.in_norm(h_pre), e], dim=-1))
        z = self.out_norm(z)
        if use_cache:
            # 推理：MTP 序列逐位置生成，缓存它的 (c_kv, k_rope)——与训练时的
            # 全序列因果注意力对齐。每次草稿调用都是合法新位置，无需回滚。
            if self._infer_cache is None or self._infer_cache.ckv.shape[0] != B:
                self._infer_cache = _LayerCache(
                    B, self.rotary.cos.shape[0], self.block.attn.kv_a.out_features - self.block.attn.dr,
                    self.block.attn.dr, h_pre.device, torch.float32)
            offset = self._infer_cache.length
            positions = torch.arange(offset, offset + T, device=h_pre.device)
            cos, sin = self.rotary(positions)
            out = self.block(z, cos, sin, layer_cache=self._infer_cache, offset=offset)
            self._infer_cache.length = offset + T
            return out
        positions = torch.arange(T, device=h_pre.device)
        cos, sin = self.rotary(positions)
        return self.block(z, cos, sin)


# --------------------------------------------------------------------------- 缓存与主模型
class _LayerCache:
    def __init__(self, batch: int, max_len: int, kv_lora: int, rope_dim: int,
                 device: torch.device, dtype: torch.dtype):
        self.length = 0
        self.ckv = torch.zeros(batch, max_len, kv_lora, device=device, dtype=dtype)
        self.krope = torch.zeros(batch, max_len, rope_dim, device=device, dtype=dtype)

    def write(self, offset: int, c_kv: torch.Tensor, k_rope: torch.Tensor) -> None:
        t = c_kv.shape[1]
        self.ckv[:, offset:offset + t] = c_kv.to(self.ckv.dtype)
        self.krope[:, offset:offset + t] = k_rope.to(self.krope.dtype)


class XingChenCache:
    """推理用 KV 缓存：每层只存 (c_kv, k_rope)，即 MLA 的压缩潜空间。"""

    def __init__(self, model: "XingChen", batch_size: int, dtype: torch.dtype = torch.float32):
        cfg = model.cfg
        device = next(model.parameters()).device
        self.length = 0
        self.layers = [_LayerCache(batch_size, cfg.max_seq_len, cfg.mla.kv_lora_rank,
                                   cfg.mla.qk_rope_head_dim, device, dtype)
                       for _ in range(cfg.n_layers)]

    def truncate(self, length: int) -> None:
        """投机解码回滚用：把有效长度缩回去，超出部分的下一次写入会覆盖。"""
        self.length = max(0, length)


class XingChen(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rotary = RotaryEmbedding(cfg.mla.qk_rope_head_dim, cfg.rope_base, cfg.max_seq_len)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = MTPModule(cfg, self.rotary) if cfg.mtp.enabled else None
        self.use_hc = cfg.hc.enabled
        self.moe_layers: List[MoEFeedForward] = [m for m in self.modules() if isinstance(m, MoEFeedForward)]
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        n_res = 2 * self.cfg.n_layers
        nn.init.normal_(self.embed.weight, std=std)
        for name, p in self.named_parameters():
            if "phi_" in name or name.endswith(".alpha_pre") or name.endswith(".alpha_post") \
                    or name.endswith(".alpha_res") or name.endswith(".b_"):
                continue  # mHC 的连接参数保持专用初始化
            if p.dim() >= 2 and "lm_head" not in name and "embed" not in name:
                nn.init.normal_(p, std=std)
        for module in self.modules():
            if isinstance(module, (MLAAttention,)):
                nn.init.normal_(module.o_proj.weight, std=std / math.sqrt(n_res))
            if isinstance(module, (Expert,)):
                nn.init.normal_(module.down_proj.weight, std=std / math.sqrt(n_res))

    # ---------------------------------------------------------------- 前向
    def forward(self, idx: torch.Tensor, cache: Optional[XingChenCache] = None):
        B, T = idx.shape
        offset = cache.length if cache is not None else 0
        assert offset + T <= self.cfg.max_seq_len, \
            f"序列超出 max_seq_len={self.cfg.max_seq_len}（当前 {offset + T}）"
        positions = torch.arange(offset, offset + T, device=idx.device)
        cos, sin = self.rotary(positions)
        x = self.embed(idx)

        if self.use_hc:
            n, C = self.cfg.hc.n_streams, self.cfg.dim
            streams = x.new_zeros(B, T, n, C)
            streams[..., 0, :] = x                      # 主流载入嵌入，其余流从零开始
            for i, layer in enumerate(self.layers):
                lc = cache.layers[i] if cache is not None else None
                streams = layer(streams, cos, sin, lc, offset)
            h = streams[..., 0, :]                      # 读主流送输出头
        else:
            for i, layer in enumerate(self.layers):
                lc = cache.layers[i] if cache is not None else None
                x = layer(x, cos, sin, lc, offset)
            h = x

        h = self.final_norm(h)
        logits = self.lm_head(h)
        if cache is not None:
            cache.length = offset + T          # 推进缓存长度（解码步能看到全部历史）
        return logits, h

    def mtp_forward(self, h_pre: torch.Tensor, next_tokens: torch.Tensor,
                    use_cache: bool = False) -> torch.Tensor:
        """MTP 分支 logits：h_pre 取主模型末端隐状态（未过 final_norm）。"""
        assert self.mtp is not None
        h = self.mtp(h_pre, self.embed(next_tokens), use_cache=use_cache)
        return self.lm_head(h)

    def reset_mtp_cache(self) -> None:
        if self.mtp is not None:
            self.mtp.reset_infer_cache()

    # ---------------------------------------------------------------- 负载均衡
    @torch.no_grad()
    def update_expert_bias(self, gamma: float) -> Optional[float]:
        """V3 无辅助损失均衡：负载偏高的专家下调偏置，偏低的上调。返回最大负载偏差。"""
        max_dev = None
        for moe in self.moe_layers:
            total = moe.expert_counts.sum()
            if total == 0:
                continue
            load = moe.expert_counts.float() / total
            target = moe.top_k / moe.n_routed
            moe.e_bias += gamma * torch.sign(target - load)
            dev = (load - target).abs().max().item()
            max_dev = dev if max_dev is None else max(max_dev, dev)
            moe.expert_counts.zero_()
        return max_dev

    def expert_load_stats(self) -> Optional[float]:
        """当前窗口内专家负载的最大相对偏差（0 = 完美均衡）。"""
        max_dev = None
        for moe in self.moe_layers:
            total = moe.expert_counts.sum()
            if total == 0:
                continue
            load = moe.expert_counts.float() / total
            dev = (load - moe.top_k / moe.n_routed).abs().max().item()
            max_dev = dev if max_dev is None else max(max_dev, dev)
        return max_dev


def param_summary(model: XingChen) -> dict:
    seen, groups = set(), {"embedding": 0.0, "attention": 0.0, "moe/ffn": 0.0, "mhc/mtp/其他": 0.0}
    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if "embed" in name or "lm_head" in name:
            key = "embedding"
        elif ".attn." in name:
            key = "attention"
        elif ".ffn." in name:
            key = "moe/ffn"
        else:
            key = "mhc/mtp/其他"
        groups[key] += p.numel()
    groups["total"] = sum(groups.values())
    return groups
