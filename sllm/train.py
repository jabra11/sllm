"""Single-GPU training. Example (size 2 on a 12GB card):

    python -m sllm.train --size 2 --asr-subset medium --max-tokens 12000 --accum 4 --grad-ckpt --out runs/s2
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from sllm.config import make_config
from sllm.data import TrainStream, collate, dev_samples, load_tokenizer, padded_vocab_size
from sllm.model import SpeechLLM
from sllm.wer import wer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--decoder", default="dense", choices=["dense", "moe"])
    p.add_argument("--asr-subset", default="small", help="Loquacious subset: small|medium|large|clean")
    p.add_argument("--stream", action="store_true", help="stream from the HF hub instead of downloading first")
    p.add_argument("--text-weight", type=float, default=1.0, help="text target tokens per ASR target token (0 = off)")
    p.add_argument("--max-tokens", type=int, default=12000, help="padded decoder tokens per micro-batch")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the encoder")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--dev-n", type=int, default=200, help="dev utterances for loss; WER on the first 100")
    p.add_argument("--out", type=Path, default=Path("runs/default"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "embed" in name or "norm" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.no_grad()
def evaluate(model, tok, dev, device, n_wer=100, bs=16):
    model.eval()
    loss_sum, n_tok = 0.0, 0
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(0, len(dev), bs):
            out = model(collate(dev[i:i + bs]).to(device))
            loss_sum += out["loss"].item() * out["num_tokens"]
            n_tok += out["num_tokens"]
        prompts = [s.__class__(s.pre, s.post[:2], s.post_loss[:2], s.wav, s.task) for s in dev[:n_wer]]
        hyps = []
        for i in range(0, len(prompts), bs):
            hyps += model.greedy_decode(collate(prompts[i:i + bs]).to(device), tok.eos_token_id)
    hyps = [tok.decode(h, skip_special_tokens=True) for h in hyps]
    model.train()
    return {"dev_loss": loss_sum / n_tok, "dev_wer": wer([s.ref for s in dev[:n_wer]], hyps),
            "example": {"ref": dev[0].ref, "hyp": hyps[0]}}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    model = SpeechLLM(make_config(args.size, args.decoder), padded_vocab_size(tok)).to(device)
    if args.grad_ckpt:
        model.enable_grad_checkpointing()
    if args.compile:
        model.encoder.compile(dynamic=True)
    n_params = {k: sum(p.numel() for p in getattr(model, k).parameters()) / 1e6
                for k in ("encoder", "adapter", "decoder")}
    print(f"params (M): {n_params}", flush=True)

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    step = 0
    ckpt_path = args.out / "last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed from step {step}", flush=True)

    stream = TrainStream(tok, args.asr_subset, args.stream, args.max_tokens, args.text_weight, seed=args.seed + step)
    loader = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                              pin_memory=True, prefetch_factor=4 if args.workers else None,
                                              persistent_workers=args.workers > 0))
    dev = dev_samples(tok, args.dev_n)
    log = open(args.out / "log.jsonl", "a")

    model.train()
    stats, t0 = defaultdict(float), time.time()
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        for _ in range(args.accum):
            batch = next(loader).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch)
            (out["loss"] / args.accum).backward()
            stats["loss"] += out["loss"].item() / args.accum
            stats["dec_tokens"] += sum(len(p) + len(q) for p, q in zip(batch.pre, batch.post))
            if batch.wav_lens is not None:
                stats["audio_sec"] += batch.wav_lens.sum().item() / 16000
            for k, v in out.items():
                if k.startswith(("loss_sum/", "tokens/", "aux_loss")):
                    stats[k] += float(v)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "lr": opt.param_groups[0]["lr"], "loss": stats["loss"] / args.log_every,
                   "grad_norm": grad_norm.item(), "audio_h_per_h": stats["audio_sec"] / dt,
                   "text_tok_per_s": stats["dec_tokens"] / dt}
            for task in ("asr", "text"):
                if stats[f"tokens/{task}"]:
                    rec[f"loss_{task}"] = stats[f"loss_sum/{task}"] / stats[f"tokens/{task}"]
            if "aux_loss" in stats:
                rec["aux_loss"] = stats["aux_loss"] / (args.log_every * args.accum)
            if device.type == "cuda":
                rec["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            stats, t0 = defaultdict(float), time.time()

        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, **evaluate(model, tok, dev, device)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(args) | {"out": str(args.out)}}, ckpt_path)


if __name__ == "__main__":
    main()
