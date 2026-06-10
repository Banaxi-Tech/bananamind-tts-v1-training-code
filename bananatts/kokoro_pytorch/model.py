from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder


class KModel(torch.nn.Module):
    """Plain PyTorch Kokoro 82M inference module for local config/checkpoints."""

    def __init__(self, config_path: str | Path, model_path: str | Path):
        super().__init__()
        with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.vocab = config["vocab"]
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        state = torch.load(Path(model_path).expanduser(), map_location="cpu", weights_only=True)
        for key, state_dict in state.items():
            if not hasattr(self, key):
                raise KeyError(f"Unexpected Kokoro state dict key: {key}")
            module = getattr(self, key)
            try:
                module.load_state_dict(state_dict)
            except RuntimeError:
                stripped = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
                module.load_state_dict(stripped, strict=False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: torch.LongTensor | None = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )

        text_mask = torch.arange(input_lengths.max(), device=input_ids.device)
        text_mask = text_mask.unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=self.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_output: bool = False,
    ) -> Output | torch.FloatTensor:
        input_ids = [self.vocab[p] for p in phonemes if p in self.vocab]
        if len(input_ids) + 2 > self.context_length:
            raise ValueError(f"Kokoro phoneme chunk is too long: {len(input_ids) + 2} > {self.context_length}")
        input_tensor = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_tensor, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
