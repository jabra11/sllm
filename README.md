# sllm

Compact speech-LLMs for "Initial Setup V1" (see `start.md`): chunked conformer encoder, frame-stacking adapter
and a randomly initialised Qwen3.5 decoder (dense or MoE), trained jointly on ASR and text.

```
16kHz wav -> log-mel (10ms, on GPU) -> conv 4x subsampling (40ms)
          -> chunked conformer (chunk 10 frames, lookahead 5)
          -> stack 2 frames + MLP (80ms per speech token)
          -> Qwen3.5 decoder:  <bos> <audio> [speech] </audio> <asr> transcript <eos>
```

| size | encoder | decoder (dense, incl. 49k-vocab embedding) |
|---|---|---|
| 1 | 20M | 86M |
| 2 | 42M | 159M |
| 3 | 105M | 411M |
| 4 | 208M | 766M |

`python -m scripts.count_params` prints these, plus the MoE variants.

## Setup (Linux, CUDA)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[cuda]"   # flash-linear-attention + causal-conv1d for fast Qwen3.5 linear-attention layers
```

Without the `cuda` extras the Gated DeltaNet layers fall back to a slow PyTorch implementation.

## Data

- ASR: [Loquacious](https://huggingface.co/datasets/speechbrain/LoquaciousSet). The subsets are `small` (250h,
  24GB), `medium` (2.5k h, 235GB), `large` (25k h, 2TB) and `clean`. Transcripts are lowercased for training.
- Text: FineWeb-Edu `sample-10BT`, always streamed, packed into 1024-token chunks.
- Mixing: `--text-weight 1.0` gives text and ASR the same number of *target* tokens.

By default the ASR subset is downloaded to the HF cache first. `--stream` reads it from the hub instead, which
is fine for `small` but is likely to become the bottleneck for bigger subsets.

## Training on a single 12GB GPU (e.g. RTX 4070 Ti), size 2

```bash
# smoke test: ~250h, a few minutes to first eval
python -m sllm.train --size 2 --asr-subset small --stream --steps 2000 --eval-every 500 --grad-ckpt --out runs/s2-smoke

# real run: Loquacious medium, ~4 epochs, ~0.7B decoder tokens
python -m sllm.train --size 2 --asr-subset medium --max-tokens 12000 --accum 4 --steps 15000 --grad-ckpt --out runs/s2-medium
```

- Watch `max_mem_gb` in the log. Raise `--max-tokens` if there is headroom, and lower it on OOM. Without
  `--grad-ckpt`, try about 4000.
- `audio_h_per_h` is hours of audio processed per wall-clock hour. If it is low while the GPU is not busy,
  raise `--workers`.
- Every `--eval-every` steps the script logs loss on the first 200 Loquacious dev utterances and greedy WER on
  the first 100, then writes `runs/<name>/last.pt`. Re-running with the same `--out` resumes from it.
- Logs go to stdout and `runs/<name>/log.jsonl`.

## Not implemented yet

ST data/task, interleaved mode, pseudo-audio, BLEU/COMET, multi-GPU.
