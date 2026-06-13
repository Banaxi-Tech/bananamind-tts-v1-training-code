from __future__ import annotations

import argparse
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio import MelSpectrogram, preemphasis_safe
from .data import prepare_ljspeech
from .models.vocoder import (
    HiFiGANGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    crop_audio_to_mel_length,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)
from .utils import count_parameters, ensure_dir, format_param_count, get_device, load_config, load_json, set_seed


class BananaVocoderDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache_root: str | Path, split: str):
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        split_path = self.cache_root / "split.json"
        if not manifest_path.exists() or not split_path.exists():
            raise FileNotFoundError(f"Missing prepared dataset in {self.cache_root}; run scripts/prepare_ljspeech.py")
        self.manifest = load_json(manifest_path)
        self.split_data = load_json(split_path)
        self.indices = list(self.split_data[split])
        self.samples = self.manifest["samples"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[self.indices[idx]]
        data = torch.load(self.cache_root / sample["path"], map_location="cpu")
        if "wav" not in data:
            raise FileNotFoundError(
                "Prepared sample does not contain waveform targets. Rebuild the cache with "
                "`python scripts/prepare_ljspeech.py --force` before training the V3 vocoder."
            )
        return {"mel": data["mel"].float(), "wav": preemphasis_safe(data["wav"].float())}


def crop_vocoder_item(item: dict[str, torch.Tensor], segment_frames: int, hop_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    mel = item["mel"]
    wav = item["wav"]
    if mel.shape[0] < segment_frames:
        mel = F.pad(mel, (0, 0, 0, segment_frames - mel.shape[0]), value=float(mel.min().item()))
    max_start = max(0, mel.shape[0] - segment_frames)
    start_frame = random.randint(0, max_start) if max_start else 0
    mel_segment = mel[start_frame : start_frame + segment_frames]
    audio_start = start_frame * hop_length
    audio_len = max(1, (segment_frames - 1) * hop_length)
    wav_segment = wav[audio_start : audio_start + audio_len]
    if wav_segment.numel() < audio_len:
        wav_segment = F.pad(wav_segment, (0, audio_len - wav_segment.numel()))
    return mel_segment.transpose(0, 1).contiguous(), wav_segment.contiguous()


def collate_vocoder_batch(batch: list[dict[str, torch.Tensor]], segment_frames: int, hop_length: int) -> dict[str, torch.Tensor]:
    mels: list[torch.Tensor] = []
    wavs: list[torch.Tensor] = []
    for item in batch:
        mel, wav = crop_vocoder_item(item, segment_frames, hop_length)
        mels.append(mel)
        wavs.append(wav)
    return {"mels": torch.stack(mels), "wavs": torch.stack(wavs)}


def batched_mels(wavs: torch.Tensor, mel_fn: MelSpectrogram) -> torch.Tensor:
    return torch.stack([mel_fn(wav).transpose(0, 1) for wav in wavs], dim=0)


def resolve_resume_checkpoint(resume: str | Path) -> Path:
    resume_str = str(resume)
    if not resume_str.startswith("hf://"):
        return Path(resume_str).expanduser()

    repo_and_file = resume_str.removeprefix("hf://")
    parts = repo_and_file.split("/", 2)
    if len(parts) != 3:
        raise ValueError(
            "HF resume URIs must look like "
            "hf://owner/repo/path/to/checkpoint.pt"
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to resume from an hf:// checkpoint URI.") from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_training_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a dict checkpoint at {path}, got {type(ckpt).__name__}")
    required = {"generator", "mpd", "msd", "optimizer_g", "optimizer_d"}
    missing = sorted(required.difference(ckpt))
    if missing:
        raise KeyError(
            f"{path} is not a full vocoder training checkpoint; missing keys: {', '.join(missing)}. "
            "Use full_vocoder.pt, not the generator-only vocoder.safetensors export, when resuming training."
        )
    return ckpt


def save_checkpoint(
    path: Path,
    generator: HiFiGANGenerator,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BananaTTS V3 HiFi-GAN vocoder.")
    parser.add_argument("--config", default="configs/bananatts_v3_hifigan.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--prepare", action="store_true", help="Prepare or refresh the dataset cache before training.")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--percent", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override vocoder_training.epochs for this run.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 1337)))
    audio_cfg = config["audio"]
    vocoder_cfg = config.get("vocoder", {})
    train_cfg = config.get("vocoder_training", config.get("training", {}))
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    cache_root = Path(config["dataset"]["cache_dir"])

    manifest_path = cache_root / "manifest.json"
    prepare_force = bool(args.force_prepare)
    needs_prepare = not manifest_path.exists()
    if args.prepare and manifest_path.exists():
        try:
            needs_prepare = not bool(load_json(manifest_path).get("contains_wav", False))
            prepare_force = prepare_force or needs_prepare
        except Exception:
            needs_prepare = True
            prepare_force = True

    if args.prepare or args.force_prepare or needs_prepare:
        prepare_ljspeech(
            config,
            dataset_name=args.dataset,
            local_path=args.local_path,
            limit=args.limit,
            percent=args.percent,
            force=prepare_force,
        )

    device = get_device()
    segment_frames = int(train_cfg.get("segment_frames", 64))
    hop_length = int(audio_cfg["hop_length"])
    train_ds = BananaVocoderDataset(cache_root, "train")
    val_ds = BananaVocoderDataset(cache_root, "val") if (cache_root / "split.json").exists() else None
    collate_fn = partial(collate_vocoder_batch, segment_frames=segment_frames, hop_length=hop_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    generator = HiFiGANGenerator(int(audio_cfg["n_mels"]), vocoder_cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        betas=(float(train_cfg.get("adam_b1", 0.8)), float(train_cfg.get("adam_b2", 0.99))),
    )
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        betas=(float(train_cfg.get("adam_b1", 0.8)), float(train_cfg.get("adam_b2", 0.99))),
    )
    start_epoch = 1
    step = 0
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        resolved_resume_path = resolve_resume_checkpoint(resume_path)
        ckpt = load_training_checkpoint(resolved_resume_path, device)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        step = int(ckpt.get("step", 0))
        print(f"Resumed V3 vocoder checkpoint: {resolved_resume_path} (next epoch {start_epoch}, step {step})")

    print(f"Generator parameters: {format_param_count(count_parameters(generator))} ({count_parameters(generator):,})")
    print(f"Discriminator parameters: {format_param_count(count_parameters(mpd) + count_parameters(msd))}")
    mel_fn = MelSpectrogram(audio_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoints_dir = ensure_dir(train_cfg.get("checkpoints_dir", "checkpoints_vocoder"))
    save_interval = int(train_cfg.get("save_interval", 1000))
    log_interval = int(train_cfg.get("log_interval", 20))
    lambda_mel = float(train_cfg.get("lambda_mel", 45.0))
    lambda_feature = float(train_cfg.get("lambda_feature", 2.0))
    target_epochs = int(train_cfg.get("epochs", 200))

    if start_epoch > target_epochs:
        print(f"Checkpoint is already past target epoch {target_epochs}; nothing to train.")
        return

    for epoch in range(start_epoch, target_epochs + 1):
        generator.train()
        mpd.train()
        msd.train()
        pbar = tqdm(train_loader, desc=f"V3 vocoder epoch {epoch}")
        for batch in pbar:
            step += 1
            mels = batch["mels"].to(device, non_blocking=True)
            real_wav = batch["wavs"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                fake_wav = crop_audio_to_mel_length(generator(mels), mels.shape[-1], hop_length)
                real_mpd = mpd(real_wav)
                fake_mpd = mpd(fake_wav.detach())
                real_msd = msd(real_wav)
                fake_msd = msd(fake_wav.detach())
                loss_d = discriminator_loss(real_mpd, fake_mpd) + discriminator_loss(real_msd, fake_msd)

            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), float(train_cfg.get("grad_clip", 10.0)))
            scaler.step(optimizer_d)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                fake_wav = crop_audio_to_mel_length(generator(mels), mels.shape[-1], hop_length)
                real_mpd = mpd(real_wav)
                fake_mpd = mpd(fake_wav)
                real_msd = msd(real_wav)
                fake_msd = msd(fake_wav)
                adv_loss = generator_loss(fake_mpd) + generator_loss(fake_msd)
                fm_loss = feature_matching_loss(real_mpd, fake_mpd) + feature_matching_loss(real_msd, fake_msd)
            fake_mel = batched_mels(fake_wav.float(), mel_fn).to(device)
            min_frames = min(fake_mel.shape[-1], mels.shape[-1])
            mel_loss = F.l1_loss(fake_mel[..., :min_frames], mels[..., :min_frames])
            loss_g = adv_loss + lambda_feature * fm_loss + lambda_mel * mel_loss

            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(train_cfg.get("grad_clip", 10.0)))
            scaler.step(optimizer_g)
            scaler.update()

            if step % log_interval == 0:
                pbar.set_postfix(
                    g=f"{float(loss_g.detach().cpu()):.3f}",
                    d=f"{float(loss_d.detach().cpu()):.3f}",
                    mel=f"{float(mel_loss.detach().cpu()):.3f}",
                )
            if step % save_interval == 0:
                save_checkpoint(
                    checkpoints_dir / "vocoder_latest.pt",
                    generator,
                    mpd,
                    msd,
                    optimizer_g,
                    optimizer_d,
                    config,
                    epoch,
                    step,
                )
                save_checkpoint(
                    checkpoints_dir / f"vocoder_step_{step}.pt",
                    generator,
                    mpd,
                    msd,
                    optimizer_g,
                    optimizer_d,
                    config,
                    epoch,
                    step,
                )

        if val_loader is not None:
            generator.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                val_mels = val_batch["mels"].to(device)
                val_fake = crop_audio_to_mel_length(generator(val_mels), val_mels.shape[-1], hop_length)
                val_fake_mel = batched_mels(val_fake, mel_fn).to(device)
                min_frames = min(val_fake_mel.shape[-1], val_mels.shape[-1])
                val_mel_loss = F.l1_loss(val_fake_mel[..., :min_frames], val_mels[..., :min_frames])
                print(f"Validation mel L1: {float(val_mel_loss.cpu()):.4f}")

        save_checkpoint(
            checkpoints_dir / "vocoder_latest.pt",
            generator,
            mpd,
            msd,
            optimizer_g,
            optimizer_d,
            config,
            epoch,
            step,
        )
        save_checkpoint(
            checkpoints_dir / f"vocoder_epoch_{epoch}.pt",
            generator,
            mpd,
            msd,
            optimizer_g,
            optimizer_d,
            config,
            epoch,
            step,
        )


if __name__ == "__main__":
    main()
