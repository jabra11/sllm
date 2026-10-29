"""Speech-LLM: log-mel -> chunked conformer -> frame-stacking adapter -> Qwen3.5 decoder (random init).

Sequence layout (prefix mode):
    ASR:  <bos> <audio> [speech embeddings] </audio> <asr> transcript <eos>
    text: <bos> text <eos>
The loss is computed only on positions marked in `post_loss` (transcript + eos, or all text tokens).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5TextConfig, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, load_balancing_loss_func

from sllm.config import DecoderConfig, ModelConfig
from sllm.conformer import ChunkedConformer
from sllm.features import LogMel

SPECIAL_TOKENS = ["<audio>", "</audio>", "<asr>"]


def build_decoder(cfg: DecoderConfig, vocab_size: int) -> nn.Module:
    common = dict(
        vocab_size=vocab_size,
        hidden_size=cfg.dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        # Gated DeltaNet layers: same head geometry as the attention layers.
        linear_num_key_heads=cfg.num_heads // 2,
        linear_num_value_heads=cfg.num_heads,
        linear_key_head_dim=cfg.head_dim,
        linear_value_head_dim=cfg.head_dim,
        tie_word_embeddings=True,
        max_position_embeddings=8192,
    )
    if cfg.kind == "dense":
        return Qwen3_5ForCausalLM(Qwen3_5TextConfig(intermediate_size=cfg.ffn_dim, **common))
    if cfg.kind == "moe":
        return Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.experts_per_tok,
            moe_intermediate_size=cfg.expert_ffn_dim,
            shared_expert_intermediate_size=cfg.shared_expert_ffn_dim,
            **common,
        ))
    raise ValueError(f"unknown decoder kind {cfg.kind!r}")


class Adapter(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, stack: int):
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(nn.Linear(enc_dim * stack, dec_dim), nn.SiLU(), nn.Linear(dec_dim, dec_dim))

    def forward(self, x, lens):
        B, T, D = x.shape
        pad = (-T) % self.stack
        x = F.pad(x, (0, 0, 0, pad)).reshape(B, (T + pad) // self.stack, D * self.stack)
        return self.net(x), (lens + self.stack - 1) // self.stack


@dataclass
class Batch:
    """Built by `sllm.data.collate`. Sample i is `pre[i] + speech(audio_idx[i]) + post[i]`."""
    wav: torch.Tensor | None  # (A, T) audio of the A speech samples in the batch
    wav_lens: torch.Tensor | None  # (A,)
    pre: list[torch.Tensor]
    post: list[torch.Tensor]
    post_loss: list[torch.Tensor]  # bool, same length as post
    audio_idx: list[int]  # row in `wav`, or -1 for text-only samples
    task: list[str]

    def to(self, device):
        if self.wav is not None:
            self.wav, self.wav_lens = self.wav.to(device, non_blocking=True), self.wav_lens.to(device)
        self.pre = [t.to(device) for t in self.pre]
        self.post = [t.to(device) for t in self.post]
        self.post_loss = [t.to(device) for t in self.post_loss]
        return self


class SpeechLLM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.logmel = LogMel(cfg.encoder.num_mels)
        self.encoder = ChunkedConformer(cfg.encoder)
        self.adapter = Adapter(cfg.encoder.dim, cfg.decoder.dim, cfg.adapter_stack)
        self.decoder = build_decoder(cfg.decoder, vocab_size)

    def enable_grad_checkpointing(self):
        self.encoder.grad_checkpointing = True
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode(self, wav, wav_lens):
        feats, feat_lens = self.logmel(wav, wav_lens)
        x, lens = self.encoder(feats, feat_lens)
        return self.adapter(x, lens)

    def _assemble(self, batch: Batch):
        """Returns inputs_embeds (B, S, D), attention_mask (B, S), targets (B, S) with -100 where no loss."""
        embed = self.decoder.get_input_embeddings()
        speech, speech_lens = self.encode(batch.wav, batch.wav_lens) if batch.wav is not None else (None, None)
        seqs, tgts = [], []
        for pre, post, post_loss, a in zip(batch.pre, batch.post, batch.post_loss, batch.audio_idx):
            parts = [embed(pre)]
            tgt = [torch.full_like(pre, -100)]
            if a >= 0:
                n = int(speech_lens[a])
                parts.append(speech[a, :n].to(parts[0].dtype))
                tgt.append(pre.new_full((n,), -100))
            parts.append(embed(post))
            tgt.append(torch.where(post_loss, post, -100))
            seqs.append(torch.cat(parts))
            tgts.append(torch.cat(tgt))
        lens = torch.tensor([len(s) for s in seqs], device=seqs[0].device)
        inputs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        targets = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=-100)
        mask = (torch.arange(inputs.size(1), device=inputs.device)[None] < lens[:, None]).long()
        return inputs, mask, targets

    def forward(self, batch: Batch, ce_chunk: int = 4096) -> dict:
        inputs, mask, targets = self._assemble(batch)
        out = self.decoder.model(inputs_embeds=inputs, attention_mask=mask,
                                 output_router_logits=self.cfg.decoder.kind == "moe", use_cache=False)
        hidden = out.last_hidden_state[:, :-1]
        targets = targets[:, 1:]
        sel = targets != -100
        h, y = hidden[sel], targets[sel]
        # Logits only for target positions, in chunks: the full (B, S, V) tensor would not fit.
        lm_head = self.decoder.lm_head
        token_losses = torch.cat([
            F.cross_entropy(lm_head(h[i:i + ce_chunk]).float(), y[i:i + ce_chunk], reduction="none")
            for i in range(0, len(y), ce_chunk)
        ])
        loss = token_losses.mean()
        result = {"loss": loss, "num_tokens": len(y)}

        # Per-task loss sums for logging.
        task_of_row = torch.tensor([hash(t) for t in batch.task], device=sel.device)
        row_of_token = sel.nonzero()[:, 0]
        for task in set(batch.task):
            m = task_of_row[row_of_token] == hash(task)
            result[f"loss_sum/{task}"] = token_losses[m].detach().sum()
            result[f"tokens/{task}"] = int(m.sum())

        if self.cfg.decoder.kind == "moe":
            aux = load_balancing_loss_func(out.router_logits, self.cfg.decoder.num_experts,
                                           self.cfg.decoder.experts_per_tok, mask)
            result["aux_loss"] = aux.detach()
            result["loss"] = loss + self.decoder.config.router_aux_loss_coef * aux
        return result

    @torch.no_grad()
    def greedy_decode(self, batch: Batch, eos_id: int, max_new_tokens: int = 200) -> list[list[int]]:
        """Decodes each sample on its own (batch size 1 avoids left-padding issues in the DeltaNet layers).
        `post` holds the prompt continuation after the audio (e.g. `</audio> <asr>`)."""
        embed = self.decoder.get_input_embeddings()
        speech, speech_lens = self.encode(batch.wav, batch.wav_lens)
        hyps = []
        for i, (pre, post, a) in enumerate(zip(batch.pre, batch.post, batch.audio_idx)):
            x = torch.cat([embed(pre), speech[a, :int(speech_lens[a])].to(embed.weight.dtype), embed(post)])[None]
            out = self.decoder(inputs_embeds=x, use_cache=True)
            hyp = []
            for _ in range(max_new_tokens):
                tok = out.logits[0, -1].argmax()
                if tok.item() == eos_id:
                    break
                hyp.append(tok.item())
                out = self.decoder(inputs_embeds=embed(tok)[None, None], past_key_values=out.past_key_values,
                                   use_cache=True)
            hyps.append(hyp)
        return hyps
