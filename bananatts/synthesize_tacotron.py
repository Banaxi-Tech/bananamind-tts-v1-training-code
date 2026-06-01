from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch

from .audio import save_wav
from .models.tacotron import TacotronLite
from .models.vocoder import GriffinLimVocoder, load_vocoder_from_config
from .text import TextTokenizer
from .utils import count_parameters, format_param_count, get_device, load_config


def load_model(config: dict[str, Any], tokenizer: TextTokenizer, checkpoint: str, device: torch.device) -> TacotronLite:
    runtime_vocoder = config.get("vocoder")
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_config = ckpt.get("config")
    if isinstance(ckpt_config, dict):
        config.clear()
        config.update(ckpt_config)
        if runtime_vocoder is not None:
            config["vocoder"] = runtime_vocoder
    model = TacotronLite(
        vocab_size=tokenizer.vocab_size,
        n_mels=int(config["audio"]["n_mels"]),
        config=config["model"],
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded Tacotron checkpoint: {checkpoint}")
    print(f"Tacotron acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
    return model


def denormalize_mel(mel: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    mel_stats = config.get("audio", {}).get("mel_stats")
    if not mel_stats or not mel_stats.get("normalized_training", False):
        return mel
    mean = float(mel_stats["mean"])
    std = max(float(mel_stats["std"]), 1e-5)
    mel = mel * std + mean
    return mel.clamp(float(mel_stats.get("min", -12.0)), float(mel_stats.get("max", 8.0)))


@torch.no_grad()
def synthesize(
    config: dict[str, Any],
    model: TacotronLite,
    tokenizer: TextTokenizer,
    text: str,
    device: torch.device,
    max_steps: int | None = None,
    stop_threshold: float | None = None,
    attention_window: int | None = None,
    normalize_wav: bool = False,
    debug: bool = False,
    vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    token_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    model_cfg = config["model"]
    output = model.infer(
        tokens,
        token_lens,
        max_steps=int(max_steps or model_cfg.get("max_decoder_steps", 1200)),
        stop_threshold=float(stop_threshold or model_cfg.get("stop_threshold", 0.55)),
        min_steps=max(20, len(token_ids) * 3),
        attention_window=int(attention_window if attention_window is not None else model_cfg.get("attention_window", 12)),
    )
    mel = denormalize_mel(output.mel_postnet[0].detach().cpu(), config)
    if debug:
        seconds = mel.shape[0] * int(config["audio"]["hop_length"]) / int(config["audio"]["sample_rate"])
        print(f"generated frames: {mel.shape[0]} (~{seconds:.2f}s)")
        print(
            "mel stats:",
            f"min={mel.min().item():.3f}",
            f"max={mel.max().item():.3f}",
            f"mean={mel.mean().item():.3f}",
        )
        align_path = output.alignments[0].argmax(dim=-1).detach().cpu()
        print(f"attention first/last: {int(align_path[0].item())} -> {int(align_path[-1].item())} / {len(token_ids) - 1}")
        print(f"attention max index reached: {int(align_path.max().item())} / {len(token_ids) - 1}")
        print(f"last stop prob: {float(torch.sigmoid(output.stop_logits[0, -1]).cpu()):.3f}")
    if vocoder is None:
        vocoder = GriffinLimVocoder(config["audio"])
    wav = vocoder(mel)
    if normalize_wav:
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech with the Tacotron-lite model.")
    parser.add_argument("--config", default="configs/bananatts_tacotron.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--attention-window", type=int, default=None)
    parser.add_argument("--normalize-wav", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vocoder", choices=["auto", "griffin-lim", "hifigan"], default="auto")
    parser.add_argument("--vocoder-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = TextTokenizer.from_config(config["text"])
    device = get_device()
    model = load_model(config, tokenizer, args.checkpoint, device)
    vocoder_type = None if args.vocoder == "auto" else args.vocoder
    vocoder = load_vocoder_from_config(config, device, checkpoint=args.vocoder_checkpoint, vocoder_type=vocoder_type)
    wav = synthesize(
        config,
        model,
        tokenizer,
        args.text,
        device,
        max_steps=args.max_steps,
        stop_threshold=args.stop_threshold,
        attention_window=args.attention_window,
        normalize_wav=args.normalize_wav,
        debug=args.debug,
        vocoder=vocoder,
    )
    save_wav(Path(args.out), wav, int(config["audio"]["sample_rate"]))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
