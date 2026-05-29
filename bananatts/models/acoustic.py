from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class DurationPredictor(nn.Module):
    def __init__(self, hidden_size: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        for layer in self.net:
            if isinstance(layer, nn.LayerNorm):
                y = layer(y.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, nn.Linear):
                y = layer(y.transpose(1, 2)).squeeze(-1)
            else:
                y = layer(y)
        return y


def make_padding_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def length_regulate(
    encoded: torch.Tensor,
    durations: torch.Tensor,
    max_duration: int = 80,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    reps: list[torch.Tensor] = []
    lengths: list[int] = []
    for batch_idx in range(encoded.shape[0]):
        dur = durations[batch_idx].clamp(min=0, max=max_duration).long()
        expanded = torch.repeat_interleave(encoded[batch_idx], dur, dim=0)
        if expanded.numel() == 0:
            expanded = encoded[batch_idx, :1]
        if max_frames is not None:
            expanded = expanded[:max_frames]
        reps.append(expanded)
        lengths.append(expanded.shape[0])
    out_len = max(lengths)
    out = encoded.new_zeros((encoded.shape[0], out_len, encoded.shape[2]))
    for i, rep in enumerate(reps):
        out[i, : rep.shape[0]] = rep
    return out, torch.tensor(lengths, device=encoded.device, dtype=torch.long)


@dataclass
class AcousticOutput:
    mel: torch.Tensor
    log_duration: torch.Tensor
    regulated_lengths: torch.Tensor


class FastSpeech2AcousticModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        config: dict[str, Any],
        pad_id: int = 0,
        special_token_ids: tuple[int, ...] | None = None,
    ):
        super().__init__()
        hidden = int(config.get("hidden_size", 192))
        heads = int(config.get("attention_heads", 4))
        ff_dim = hidden * int(config.get("ff_multiplier", 4))
        dropout = float(config.get("dropout", 0.1))
        self.pad_id = pad_id
        self.special_token_ids = tuple(special_token_ids or (0, 2, 3))
        self.max_duration = int(config.get("max_duration", 80))
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.get("encoder_layers", 4)))
        self.duration_predictor = DurationPredictor(
            hidden,
            int(config.get("duration_conv_channels", hidden)),
            int(config.get("duration_kernel_size", 3)),
            dropout,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(config.get("decoder_layers", 4)))
        self.mel_proj = nn.Linear(hidden, n_mels)

    def forward(
        self,
        tokens: torch.Tensor,
        token_lens: torch.Tensor,
        durations: torch.Tensor | None = None,
        max_frames: int | None = None,
        duration_scale: float = 1.0,
    ) -> AcousticOutput:
        key_padding = make_padding_mask(token_lens, tokens.shape[1])
        special_mask = torch.zeros_like(key_padding)
        for token_id in self.special_token_ids:
            special_mask |= tokens == token_id
        duration_mask = key_padding | special_mask
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos(x)
        encoded = self.encoder(x, src_key_padding_mask=key_padding)
        log_duration = self.duration_predictor(encoded).masked_fill(duration_mask, 0.0)
        if durations is None:
            duration_values = (torch.exp(log_duration) - 1.0) * float(duration_scale)
            pred_durations = torch.clamp(torch.round(duration_values), min=0, max=self.max_duration).long()
            content_mask = ~duration_mask
            pred_durations = pred_durations.masked_fill(content_mask & pred_durations.eq(0), 1)
            pred_durations = pred_durations.masked_fill(duration_mask, 0)
            durations = pred_durations
        regulated, regulated_lens = length_regulate(encoded, durations, self.max_duration, max_frames=max_frames)
        dec_padding = make_padding_mask(regulated_lens, regulated.shape[1])
        regulated = self.pos(regulated)
        decoded = self.decoder(regulated, src_key_padding_mask=dec_padding)
        mel = self.mel_proj(decoded)
        return AcousticOutput(mel=mel, log_duration=log_duration, regulated_lengths=regulated_lens)


def acoustic_loss(
    output: AcousticOutput,
    target_mel: torch.Tensor,
    target_durations: torch.Tensor,
    mel_lens: torch.Tensor,
    token_lens: torch.Tensor,
    mel_weight: float = 1.0,
    duration_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_mel = min(output.mel.shape[1], target_mel.shape[1])
    mel_pred = output.mel[:, :max_mel]
    mel_tgt = target_mel[:, :max_mel]
    mel_mask = ~make_padding_mask(mel_lens.clamp(max=max_mel), max_mel)
    mel_loss = F.l1_loss(mel_pred[mel_mask], mel_tgt[mel_mask])

    max_tok = output.log_duration.shape[1]
    tok_mask = ~make_padding_mask(token_lens, max_tok)
    target_log_dur = torch.log(target_durations[:, :max_tok].float().clamp_min(0) + 1.0)
    dur_loss = F.mse_loss(output.log_duration[tok_mask], target_log_dur[tok_mask])
    loss = mel_weight * mel_loss + duration_weight * dur_loss
    return loss, {"mel_loss": float(mel_loss.detach().cpu()), "duration_loss": float(dur_loss.detach().cpu())}
