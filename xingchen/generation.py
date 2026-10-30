# -*- coding: utf-8 -*-
"""星辰 · 生成（采样 + MLA 缓存解码）

标准采样套件：temperature / top-k / top-p / repetition penalty。
预填充一次 prompt，之后逐 token 走 MLA 压缩缓存。
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from .model import XingChen, XingChenCache


def _sample_token(last_logits: torch.Tensor, temperature: float, top_k: int,
                  top_p: float, banned: Optional[set] = None) -> int:
    """从最后一位置的 logits 里按 temperature/top-k/top-p 采一个 token。
    接受 (B, P, V)（取最后位置）与 (B, V)（已是单位置）两种形状。"""
    last = last_logits[0]
    if last.dim() == 2:                                  # (P, V) → 取最后一个位置
        last = last[-1]
    last = last.float().clone()
    if banned:
        for t in banned:
            last[t] = float("-inf")
    if temperature <= 0:
        return int(last.argmax())
    last = last / temperature
    if top_k > 0:
        kth = torch.topk(last, min(top_k, last.numel())).values[-1]
        last[last < kth] = float("-inf")
    if 0.0 < top_p < 1.0:
        probs = F.softmax(last, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > top_p
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        last[sorted_idx[cutoff]] = float("-inf")
    probs = F.softmax(last, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def generate_speculative(
    model: XingChen,
    prompt_ids: List[int],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_id: Optional[int] = None,
    callback: Optional[Callable[[int], None]] = None,
) -> tuple:
    """MTP 投机解码（深度 1 草稿）。

    主模型一次前向处理 [已采样 token, 草稿 token] 两个位置；MTP 分支负责草稿，
    主模型的分布负责验证。贪心（temperature=0）时输出与 generate(temperature=0)
    完全一致。MTP 分支推理为无缓存近似（训练时它看得到全序列上下文），
    实际收益以接受率为准。
    返回 (ids, stats)；stats 含 drafts/accepted/steps。
    """
    import random as _random
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    max_len = model.cfg.max_seq_len
    room = max_len - len(prompt_ids) - 3
    if room <= 0:
        raise ValueError(f"prompt 过长：{len(prompt_ids)} ≥ max_seq_len {max_len}")
    max_new_tokens = min(max_new_tokens, room)

    ids = list(prompt_ids)
    cache = XingChenCache(model, batch_size=1, dtype=torch.float32)
    model.reset_mtp_cache()
    logits, h = model(torch.tensor([ids], device=device), cache=cache)

    stats = {"drafts": 0, "accepted": 0, "steps": 0}

    def draft_and_verify(h_prev, pending_a, budget_after_commit):
        """对 pending_a 做草稿+验证。返回 (本迭代提交的 tokens, 下一状态)。"""
        nonlocal cache
        m_logits = model.mtp_forward(h_prev, torch.tensor([[pending_a]], device=device),
                                     use_cache=True)
        if temperature <= 0:
            d = int(m_logits[0, -1].argmax())
        else:
            d = _sample_token(m_logits, temperature, top_k, top_p)
        # 主模型一次前向同时处理 [pending_a, d] 两个位置
        logits2, h2 = model(torch.tensor([[pending_a, d]], device=device), cache=cache)
        target_after_a = logits2[:, 0]                      # a 之后的目标分布 (1,V)
        if temperature <= 0:
            accepted = (d == int(target_after_a[0].argmax()))
        else:
            p_t = F.softmax(target_after_a[0].float(), dim=-1)
            p_d = F.softmax(m_logits[0, -1].float(), dim=-1)
            accepted = _random.random() < min(1.0, (p_t[d] + 1e-12) / (p_d[d] + 1e-12))
        committed = [pending_a]
        if accepted:
            committed.append(d)
        return committed, logits2, h2, accepted, target_after_a

    pending_a = _sample_token(logits, temperature, top_k, top_p)
    h_prev = h[:, -1:]
    budget = max_new_tokens
    while budget > 0:
        stats["steps"] += 1
        if budget == 1:
            ids.append(pending_a)                           # 预算只剩 1，不投机
            if callback is not None:
                callback(pending_a)
            break
        committed, logits2, h2, accepted, target_after_a = draft_and_verify(
            h_prev, pending_a, budget - 1)
        stats["drafts"] += 1
        stats["accepted"] += int(accepted)
        for t in committed:
            ids.append(t)
            if callback is not None:
                callback(t)
        budget -= len(committed)
        if eos_id is not None and any(t == eos_id for t in committed):
            break
        if accepted:
            pending_a = _sample_token(logits2[:, 1], temperature, top_k, top_p)
            h_prev = h2[:, 1:2]
        else:
            # 拒绝：撤掉草稿 d，提交目标自己的选择，并重新前向校正后的 token
            correct = _sample_token(target_after_a, temperature, top_k, top_p)
            cache.truncate(len(ids))                        # len(ids)=P+1，撤掉 d
            ids.append(correct)
            if callback is not None:
                callback(correct)
            budget -= 1
            if eos_id is not None and correct == eos_id:
                break
            logits3, h3 = model(torch.tensor([[correct]], device=device), cache=cache)
            pending_a = _sample_token(logits3, temperature, top_k, top_p)
            h_prev = h3[:, -1:]

    if was_training:
        model.train()
    return ids, stats


@torch.no_grad()
def generate(
    model: XingChen,
    prompt_ids: List[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
    eos_id: Optional[int] = None,
    callback: Optional[Callable[[int], None]] = None,
) -> List[int]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    max_len = model.cfg.max_seq_len
    room = max_len - len(prompt_ids) - 1
    if room <= 0:
        raise ValueError(f"prompt 过长：{len(prompt_ids)} ≥ max_seq_len {max_len}")
    max_new_tokens = min(max_new_tokens, room)

    ids = list(prompt_ids)
    cache = XingChenCache(model, batch_size=1, dtype=torch.float32)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache)

    for _ in range(max_new_tokens):
        last = logits[0, -1].float().clone()
        if repetition_penalty != 1.0:
            for t in set(ids):
                last[t] = last[t] / repetition_penalty if last[t] > 0 else last[t] * repetition_penalty
        if no_repeat_ngram_size > 0 and len(ids) >= no_repeat_ngram_size - 1:
            # 禁止复现已出现过的 n-gram 后继，掐断复读循环（HF generate 同思路）
            w = no_repeat_ngram_size - 1
            prefix = tuple(ids[-w:]) if w > 0 else ()
            banned = {ids[i + w] for i in range(len(ids) - w) if tuple(ids[i:i + w]) == prefix}
            if banned:
                last[list(banned)] = float("-inf")

        if temperature <= 0:
            tok = int(last.argmax())
        else:
            last = last / temperature
            if top_k > 0:
                kth = torch.topk(last, min(top_k, last.numel())).values[-1]
                last[last < kth] = float("-inf")
            if 0.0 < top_p < 1.0:
                probs = F.softmax(last, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cum > top_p
                cutoff[1:] = cutoff[:-1].clone()      # 保住第一个越过阈值的 token
                cutoff[0] = False
                last[sorted_idx[cutoff]] = float("-inf")
            probs = F.softmax(last, dim=-1)
            tok = int(torch.multinomial(probs, num_samples=1))

        if callback is not None:
            callback(tok)
        if eos_id is not None and tok == eos_id:
            break
        ids.append(tok)

        logits, _ = model(torch.tensor([[tok]], device=device), cache=cache)

    if was_training:
        model.train()
    return ids
