"""Log-mel features computed on the GPU, so data loader workers only decode audio."""

import math

import torch
import torch.nn as nn


def _mel_filterbank(num_mels: int, n_fft: int, sample_rate: int, f_min: float = 20.0) -> torch.Tensor:
    def hz_to_mel(f):
        return 1127.0 * math.log1p(f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (math.exp(m / 1127.0) - 1.0)

    f_max = sample_rate / 2
    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    hz_pts = torch.tensor([mel_to_hz(m_min + (m_max - m_min) * i / (num_mels + 1)) for i in range(num_mels + 2)])
    fft_freqs = torch.linspace(0, f_max, n_fft // 2 + 1)
    lower, center, upper = hz_pts[:-2, None], hz_pts[1:-1, None], hz_pts[2:, None]
    up = (fft_freqs - lower) / (center - lower)
    down = (upper - fft_freqs) / (upper - center)
    return torch.clamp(torch.minimum(up, down), min=0.0)  # (num_mels, n_fft // 2 + 1)


class LogMel(nn.Module):
    """25ms window, 10ms shift, 16kHz input."""

    def __init__(self, num_mels: int = 80, sample_rate: int = 16000):
        super().__init__()
        self.win_length = sample_rate * 25 // 1000
        self.hop_length = sample_rate * 10 // 1000
        self.n_fft = 512
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("fbank", _mel_filterbank(num_mels, self.n_fft, sample_rate), persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav: (B, T) float32, wav_lens: (B,) samples -> feats (B, F, num_mels), feat_lens (B,)"""
        spec = torch.stft(
            wav.float(), self.n_fft, self.hop_length, self.win_length, self.window,
            center=False, return_complex=True,
        )
        power = spec.abs().pow(2)  # (B, n_fft//2+1, F)
        feats = torch.log(torch.matmul(self.fbank, power).clamp(min=1e-10)).transpose(1, 2)
        feat_lens = (wav_lens - self.win_length) // self.hop_length + 1
        # Per-utterance mean/variance normalisation over valid frames.
        mask = (torch.arange(feats.size(1), device=feats.device)[None] < feat_lens[:, None]).unsqueeze(-1)
        n = feat_lens.clamp(min=1)[:, None, None]
        mean = (feats * mask).sum(1, keepdim=True) / n
        std = (((feats - mean) ** 2 * mask).sum(1, keepdim=True) / n).sqrt().clamp(min=1e-5)
        return ((feats - mean) / std) * mask, feat_lens
