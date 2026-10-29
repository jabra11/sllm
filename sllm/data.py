"""Streaming data pipeline: Loquacious (ASR) + FineWeb-Edu (text), mixed by target tokens, dynamically batched.

Each DataLoader worker reads its own subset of shards (handled by `datasets`), mixes the sources, buckets
samples by length and yields ready-to-use `Batch` objects.
"""

import io
import random
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

from sllm.model import SPECIAL_TOKENS, Batch

SAMPLE_RATE = 16000
SPEECH_TOKENS_PER_SEC = 12.5  # 80ms per speech token after the adapter


def load_tokenizer(name: str = "HuggingFaceTB/SmolLM2-135M"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def padded_vocab_size(tok, multiple: int = 64) -> int:
    return (len(tok) + multiple - 1) // multiple * multiple


@dataclass
class Sample:
    pre: list[int]
    post: list[int]
    post_loss: list[bool]
    wav: np.ndarray | None
    task: str
    ref: str = ""

    @property
    def length(self) -> int:
        n_speech = 0 if self.wav is None else int(len(self.wav) / SAMPLE_RATE * SPEECH_TOKENS_PER_SEC) + 1
        return len(self.pre) + n_speech + len(self.post)

    @property
    def num_targets(self) -> int:
        return sum(self.post_loss)


class Prompts:
    def __init__(self, tok):
        self.tok = tok
        self.bos = self.eos = tok.eos_token_id
        self.audio, self.audio_end, self.asr_task = tok.convert_tokens_to_ids(SPECIAL_TOKENS)

    def asr(self, wav: np.ndarray, text: str | None) -> Sample:
        post = [self.audio_end, self.asr_task]
        if text is None:  # decoding: prompt only
            return Sample([self.bos, self.audio], post, [False] * 2, wav, "asr")
        ids = self.tok.encode(text.lower(), add_special_tokens=False) + [self.eos]
        return Sample([self.bos, self.audio], post + ids, [False] * 2 + [True] * len(ids), wav, "asr", ref=text)

    def text(self, ids: list[int]) -> Sample:
        return Sample([self.bos], ids, [True] * len(ids), None, "text")


def decode_audio(entry: dict) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz, got {sr}"
    return wav


def _forever(make_iter):
    epoch = 0
    while True:
        yield from make_iter(epoch)
        epoch += 1


def asr_samples(prompts: Prompts, subset: str, split: str, streaming: bool, max_dur: float, seed: int):
    ds = load_dataset("speechbrain/LoquaciousSet", subset, split=split, streaming=streaming)
    ds = ds.cast_column("wav", Audio(decode=False))
    if not streaming:
        ds = ds.to_iterable_dataset(num_shards=256)
    ds = ds.filter(lambda d: d <= max_dur, input_columns="duration").shuffle(seed=seed, buffer_size=5000)

    def run(epoch):
        ds.set_epoch(epoch)
        for ex in ds:
            yield prompts.asr(decode_audio(ex["wav"]), ex["text"])

    return _forever(run)


def text_samples(prompts: Prompts, seq_len: int, seed: int):
    """FineWeb-Edu, documents concatenated with eos and cut into chunks of `seq_len` tokens."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    def run(epoch):
        ds.set_epoch(epoch)
        buf = []
        for ex in ds:
            buf += prompts.tok.encode(ex["text"], add_special_tokens=False) + [prompts.eos]
            while len(buf) >= seq_len:
                yield prompts.text(buf[:seq_len])
                buf = buf[seq_len:]

    return _forever(run)


def mix_by_tokens(sources: dict, weights: dict):
    """Always draws from the source furthest below its share of target tokens."""
    seen = {k: 0 for k in sources}
    while True:
        k = min(sources, key=lambda k: seen[k] / weights[k])
        s = next(sources[k])
        seen[k] += s.num_targets
        yield s


def collate(samples: list[Sample]) -> Batch:
    wavs = [s.wav for s in samples if s.wav is not None]
    audio_idx, a = [], 0
    for s in samples:
        audio_idx.append(a if s.wav is not None else -1)
        a += s.wav is not None
    wav = wav_lens = None
    if wavs:
        wav_lens = torch.tensor([len(w) for w in wavs])
        wav = torch.zeros(len(wavs), int(wav_lens.max()))
        for i, w in enumerate(wavs):
            wav[i, :len(w)] = torch.from_numpy(w)
    return Batch(
        wav=wav, wav_lens=wav_lens,
        pre=[torch.tensor(s.pre) for s in samples],
        post=[torch.tensor(s.post) for s in samples],
        post_loss=[torch.tensor(s.post_loss) for s in samples],
        audio_idx=audio_idx,
        task=[s.task for s in samples],
    )


def bucket_batches(samples, max_tokens: int, buffer: int = 2000, seed: int = 0):
    """Fills a buffer, sorts by length and cuts batches whose padded size (n * max_len) stays below max_tokens."""
    rng = random.Random(seed)
    while True:
        buf = sorted((next(samples) for _ in range(buffer)), key=lambda s: s.length)
        batches, cur = [], []
        for s in buf:
            if cur and (len(cur) + 1) * s.length > max_tokens:
                batches.append(cur)
                cur = []
            cur.append(s)
        rng.shuffle(batches)  # the last, partial batch is dropped
        yield from (collate(b) for b in batches)


class TrainStream(torch.utils.data.IterableDataset):
    def __init__(self, tok, asr_subset: str, streaming: bool, max_tokens: int, text_weight: float,
                 text_seq_len: int = 1024, max_dur: float = 30.0, seed: int = 0):
        self.tok = tok
        self.kw = dict(asr_subset=asr_subset, streaming=streaming, max_tokens=max_tokens, text_weight=text_weight,
                       text_seq_len=text_seq_len, max_dur=max_dur, seed=seed)

    def __iter__(self):
        kw = self.kw
        info = torch.utils.data.get_worker_info()
        seed = kw["seed"] + (info.id if info else 0)
        prompts = Prompts(self.tok)
        sources = {"asr": asr_samples(prompts, kw["asr_subset"], "train", kw["streaming"], kw["max_dur"], seed)}
        weights = {"asr": 1.0}
        if kw["text_weight"] > 0:
            sources["text"] = text_samples(prompts, kw["text_seq_len"], seed)
            weights["text"] = kw["text_weight"]
        yield from bucket_batches(mix_by_tokens(sources, weights), kw["max_tokens"], seed=seed)


def dev_samples(tok, n: int, split: str = "dev") -> list[Sample]:
    """First n utterances of the Loquacious dev set, with transcripts (for loss) in `ref`."""
    ds = load_dataset("speechbrain/LoquaciousSet", "small", split=split, streaming=True)
    ds = ds.cast_column("wav", Audio(decode=False))
    prompts = Prompts(tok)
    return [prompts.asr(decode_audio(ex["wav"]), ex["text"]) for ex in ds.take(n)]
