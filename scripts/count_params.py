"""Prints encoder / decoder parameter counts for all size presets.

    python -m scripts.count_params
"""

import torch

from sllm.config import SIZES, make_config
from sllm.model import SpeechLLM


def count(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def main(vocab_size: int = 49216):
    print(f"{'size':>4} {'decoder':>7} {'encoder':>8} {'adapter':>8} {'dec non-emb':>12} {'dec total':>10}")
    for size in SIZES:
        for kind in ("dense", "moe"):
            with torch.device("meta"):
                m = SpeechLLM(make_config(size, kind), vocab_size)
            emb = m.decoder.get_input_embeddings().weight.numel() / 1e6
            dec = count(m.decoder)
            print(f"{size:>4} {kind:>7} {count(m.encoder):>7.1f}M {count(m.adapter):>7.1f}M "
                  f"{dec - emb:>11.1f}M {dec:>9.1f}M")


if __name__ == "__main__":
    main()
