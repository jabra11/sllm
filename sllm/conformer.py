"""Chunked conformer encoder.

After 4x subsampling the sequence is split into chunks of `chunk_size` frames. Every chunk gets its own
copy of the next `right_context` frames (lookahead), so each extended chunk is (chunk_size + right_context)
frames long. All extended chunks are processed side by side:

- attention: an extended chunk attends to itself and to the main (non-lookahead) frames of all previous
  chunks; it never sees the lookahead copies of other chunks, so future context does not leak across layers
- convolution: applied within each extended chunk only

After the last layer the lookahead copies are dropped. This matches what a streaming decoder would compute
with full left-context caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sllm.config import EncoderConfig


class Subsample4x(nn.Module):
    def __init__(self, num_mels: int, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.out = nn.Linear(dim * ((num_mels + 3) // 4), dim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))  # (B, D, T/4, F/4)
        x = self.out(x.permute(0, 2, 1, 3).flatten(2))
        for _ in range(2):
            lens = (lens - 1) // 2 + 1
        return x, lens


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim),
                             persistent=False)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * mult, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, mask, cos, sin):
        B, S, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, S, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return F.dropout(self.out(y.transpose(1, 2).reshape(B, S, D)), self.dropout, self.training)


class ConvModule(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 2 * dim)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, chunk_mask):
        """x: (B*N, W, D) one row per extended chunk; chunk_mask: (B*N, W, 1) valid frames."""
        x = F.glu(self.pw1(self.norm(x)), dim=-1) * chunk_mask
        x = self.dw(x.transpose(1, 2))
        x = F.silu(self.bn(x)).transpose(1, 2)
        return F.dropout(self.pw2(x), self.dropout, self.training)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.ff1 = FeedForward(cfg.dim, cfg.ff_mult, cfg.dropout)
        self.attn = SelfAttention(cfg.dim, cfg.num_heads, cfg.dropout)
        self.conv = ConvModule(cfg.dim, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(cfg.dim, cfg.ff_mult, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x, attn_mask, chunk_mask, cos, sin, num_chunks):
        """x: (B, N*W, D)"""
        B, S, D = x.shape
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, attn_mask, cos, sin)
        x = x + self.conv(x.reshape(B * num_chunks, S // num_chunks, D), chunk_mask).reshape(B, S, D)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ChunkedConformer(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = Subsample4x(cfg.num_mels, cfg.dim)
        self.rope = RotaryEmbedding(cfg.dim // cfg.num_heads)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.num_layers))
        self.grad_checkpointing = False

    def _chunk_layout(self, T: int, device):
        """Returns source frame index per extended position (N, W), lookahead flag (W,), N."""
        C, R = self.cfg.chunk_size, self.cfg.right_context
        N = (T + C - 1) // C
        idx = torch.arange(N, device=device)[:, None] * C + torch.arange(C + R, device=device)[None]
        is_lookahead = torch.arange(C + R, device=device) >= C
        return idx, is_lookahead, N

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lens = self.subsample(feats, feat_lens)
        B, T, D = x.shape
        C = self.cfg.chunk_size
        idx, is_lookahead, N = self._chunk_layout(T, x.device)
        W = idx.size(1)

        valid = idx[None] < lens[:, None, None]  # (B, N, W)
        x = F.pad(x, (0, 0, 0, N * C + W - T))[:, idx.flatten()]  # (B, N*W, D)

        # Query (n, i) may attend key (m, j) if m == n, or m < n and j is a main frame.
        chunk_of = torch.arange(N, device=x.device).repeat_interleave(W)
        key_main = ~is_lookahead.repeat(N)
        allowed = (chunk_of[:, None] == chunk_of[None]) | ((chunk_of[None] < chunk_of[:, None]) & key_main[None])
        attn_mask = allowed[None, None] & valid.view(B, 1, 1, N * W)
        # Fully padded rows would produce NaNs in SDPA; let them attend to themselves.
        attn_mask = attn_mask | torch.eye(N * W, dtype=torch.bool, device=x.device)[None, None]
        chunk_mask = valid.view(B * N, W, 1).to(x.dtype)

        cos, sin = self.rope(idx.flatten())
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, attn_mask, chunk_mask, cos, sin, N, use_reentrant=False)
            else:
                x = block(x, attn_mask, chunk_mask, cos, sin, N)

        x = x.view(B, N, W, D)[:, :, :C].reshape(B, N * C, D)[:, :T]
        return x, lens
