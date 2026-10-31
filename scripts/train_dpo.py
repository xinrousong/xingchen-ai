# -*- coding: utf-8 -*-
"""M1.6 第二步：DPO（Direct Preference Optimization）拒绝采样对齐。

思路：不训练奖励模型，直接用「策略 vs 参考」的隐式奖励做对比学习：
    loss = -logsigmoid( β * [ (logp_θ(y+) - logp_ref(y+)) - (logp_θ(y-) - logp_ref(y-)) ] )
参考模型 = 训练前的 SFT 模型（冻结），所以只需要一份偏好对数据。

用法：
    python scripts/train_dpo.py --init checkpoints-base-sft/latest.pt \
        --pairs data/dpo/pairs.jsonl --out_dir checkpoints-base-dpo --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xingchen.config import load_config  # noqa: E402
from xingchen.model import XingChen  # noqa: E402
from xingchen.tokenizer import ByteBPETokenizer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def seq_logprob(model, prompt_ids: list[int], ans_ids: list[int], device, autocast: bool) -> torch.Tensor:
    """回答段的对数概率之和。"""
    ids = torch.tensor([prompt_ids + ans_ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits, _ = model(ids[:, :-1])
    logits = logits.float()
    start = len(prompt_ids)
    tgt = ids[0, start:]
    if tgt.numel() == 0:
        return torch.zeros((), device=device)
    lp = torch.log_softmax(logits, dim=-1)[0, start - 1:start - 1 + tgt.numel()]
    return lp.gather(-1, tgt.unsqueeze(-1)).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "checkpoints-base-sft" / "latest.pt"))
    ap.add_argument("--pairs", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    ap.add_argument("--out_dir", default=str(ROOT / "checkpoints-base-dpo"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "base-wiki.yaml"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=508)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pairs:
        raise SystemExit("偏好对为空，先跑 build_dpo_pairs.py")
    print(f"[dpo] 偏好对 {len(pairs)} 条，beta={args.beta}, lr={args.lr}, epochs={args.epochs}", flush=True)

    cfg = load_config(args.config)
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    tok = ByteBPETokenizer.load(ckpt.get("tokenizer_path") or str(ROOT / "data/wiki/tokenizer.json"))

    policy = XingChen(cfg.model).to(args.device)
    policy.load_state_dict(ckpt["model"])
    policy.train()

    ref = deepcopy(XingChen(cfg.model)).to(args.device).eval()
    ref.load_state_dict(ckpt["model"])
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    def jlog(rec: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jlog({"type": "meta", "init": Path(args.init).name, "pairs": len(pairs),
          "beta": args.beta, "lr": args.lr, "epochs": args.epochs})

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for row in pairs:
            p_ids = tok.encode(row["prompt"])
            # eos 是特殊 token，encode() 不会把它当文本识别，必须手动追加 id
            c_ids = tok.encode(row["chosen"]) + [tok.eos_id]
            r_ids = tok.encode(row["rejected"]) + [tok.eos_id]
            room = args.max_len - len(p_ids)
            if room < 8:
                continue
            c_ids, r_ids = c_ids[:room], r_ids[:room]

            lp_c = seq_logprob(policy, p_ids, c_ids, args.device, True)
            lp_r = seq_logprob(policy, p_ids, r_ids, args.device, True)
            with torch.no_grad():
                ref_c = seq_logprob(ref, p_ids, c_ids, args.device, True)
                ref_r = seq_logprob(ref, p_ids, r_ids, args.device, True)

            margin = (lp_c - ref_c) - (lp_r - ref_r)
            loss = -F.logsigmoid(args.beta * margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 20 == 0:
                acc = float((margin > 0).item())
                print(f"[dpo] step {step:5d} | loss {loss.item():.4f} | margin {margin.item():+.3f} "
                      f"| 偏好正确 {acc:.0f} | {time.time()-t0:.0f}s", flush=True)
                jlog({"type": "train", "step": step, "loss": round(loss.item(), 4),
                      "margin": round(margin.item(), 4)})

            if step % args.save_every == 0:
                torch.save({"model": policy.state_dict(), "iter": step, "config": cfg.to_dict(),
                            "tokenizer_path": str(ckpt.get("tokenizer_path") or "")},
                           out_dir / f"dpo_{step:05d}.pt")
                torch.save({"model": policy.state_dict(), "iter": step, "config": cfg.to_dict(),
                            "tokenizer_path": str(ckpt.get("tokenizer_path") or "")},
                           out_dir / "latest.pt")
                print(f"[dpo] 已保存 dpo_{step:05d}.pt", flush=True)

    torch.save({"model": policy.state_dict(), "iter": step, "config": cfg.to_dict(),
                "tokenizer_path": str(ckpt.get("tokenizer_path") or "")}, out_dir / "latest.pt")
    jlog({"type": "done", "steps": step})
    print(f"[dpo] 完成 {step} 步，最终检查点 {out_dir/'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
