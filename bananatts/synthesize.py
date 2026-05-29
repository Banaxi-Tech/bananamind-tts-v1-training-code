from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .audio import save_wav
from .models.acoustic import FastSpeech2AcousticModel
from .models.vocoder import GriffinLimVocoder
from .text import TextTokenizer
from .utils import count_parameters, format_param_count, get_device, load_config


def load_acoustic(
    config: dict[str, Any],
    tokenizer: TextTokenizer,
    checkpoint: str | None,
    device: torch.device,
) -> FastSpeech2AcousticModel:
    model = FastSpeech2AcousticModel(
        vocab_size=tokenizer.vocab_size,
        n_mels=int(config["audio"]["n_mels"]),
        config=config["model"],
        pad_id=tokenizer.pad_id,
        special_token_ids=(tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id),
    ).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        ckpt_config = ckpt.get("config")
        if isinstance(ckpt_config, dict):
            config.clear()
            config.update(ckpt_config)
        print(f"Loaded acoustic checkpoint: {checkpoint}")
    else:
        print("No checkpoint provided; synthesizing with random acoustic weights for pipeline debugging.")
    model.eval()
    print(f"Acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
    return model


@torch.no_grad()
def synthesize_text(
    config: dict[str, Any],
    model: FastSpeech2AcousticModel,
    tokenizer: TextTokenizer,
    text: str,
    device: torch.device,
    duration_scale: float = 1.0,
    fixed_duration: int | None = None,
    min_duration: int = 1,
    wav_gain: float = 1.0,
    normalize_wav: bool = False,
    debug: bool = False,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    token_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    durations = None
    special_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
    if fixed_duration is not None:
        durations = torch.tensor(
            [[0 if token_id in special_ids else max(1, fixed_duration) for token_id in token_ids]],
            dtype=torch.long,
            device=device,
        )
    output = model(tokens, token_lens, durations=durations, duration_scale=duration_scale)
    if durations is None and min_duration > 1:
        pred = torch.clamp(torch.round(torch.exp(output.log_duration) - 1.0), min=0).long()
        special_mask = torch.zeros_like(pred, dtype=torch.bool)
        for token_id in special_ids:
            special_mask |= tokens == token_id
        pred = pred.masked_fill((~special_mask) & pred.lt(min_duration), min_duration)
        pred = pred.masked_fill(special_mask, 0)
        output = model(tokens, token_lens, durations=pred)
        durations = pred
    if debug:
        if durations is None:
            shown = torch.clamp(torch.round((torch.exp(output.log_duration[0]) - 1.0) * duration_scale), min=0).long()
            for i, token_id in enumerate(token_ids):
                if token_id in special_ids:
                    shown[i] = 0
        else:
            shown = durations[0].detach().cpu()
        seconds = int(shown.sum().item()) * int(config["audio"]["hop_length"]) / int(config["audio"]["sample_rate"])
        symbols = [tokenizer.symbols[token_id] if 0 <= token_id < len(tokenizer.symbols) else "?" for token_id in token_ids]
        print("tokens:", symbols)
        print("durations:", shown.tolist())
        print(f"predicted frames: {int(shown.sum().item())} (~{seconds:.2f}s)")
    mel = output.mel[0, : output.regulated_lengths[0]].detach().cpu()
    mel_stats = config.get("audio", {}).get("mel_stats")
    if mel_stats and mel_stats.get("normalized_training", False):
        mean = float(mel_stats["mean"])
        std = max(float(mel_stats["std"]), 1e-5)
        mel = mel * std + mean
        mel_min = float(mel_stats.get("min", -12.0))
        mel_max = float(mel_stats.get("max", 8.0))
        mel = mel.clamp(mel_min, mel_max)
    if debug:
        print(
            "mel stats:",
            f"shape={tuple(mel.shape)}",
            f"min={mel.min().item():.3f}",
            f"max={mel.max().item():.3f}",
            f"mean={mel.mean().item():.3f}",
        )
    vocoder = GriffinLimVocoder(config["audio"])
    wav = vocoder(mel)
    if wav_gain != 1.0:
        wav = wav * float(wav_gain)
    if normalize_wav:
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
    if debug:
        print(f"wav peak: {wav.abs().max().item():.4f}")
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech from text.")
    parser.add_argument("--config", default="configs/bananatts_20m.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--fixed-duration", type=int, default=None, help="Bypass duration predictor; frames per non-special token")
    parser.add_argument("--min-duration", type=int, default=1, help="Minimum frames per non-special token during predicted-duration synthesis")
    parser.add_argument("--wav-gain", type=float, default=1.0)
    parser.add_argument("--normalize-wav", action="store_true")
    parser.add_argument("--debug-durations", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = TextTokenizer.from_config(config["text"])
    device = get_device()
    model = load_acoustic(config, tokenizer, args.checkpoint, device)
    wav = synthesize_text(
        config,
        model,
        tokenizer,
        args.text,
        device,
        duration_scale=args.duration_scale,
        fixed_duration=args.fixed_duration,
        min_duration=args.min_duration,
        wav_gain=args.wav_gain,
        normalize_wav=args.normalize_wav,
        debug=args.debug_durations,
    )
    save_wav(Path(args.out), wav, int(config["audio"]["sample_rate"]))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
