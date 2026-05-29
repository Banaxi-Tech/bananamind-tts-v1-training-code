from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BananaTTSDataset, collate_batch, prepare_ljspeech
from .models.tacotron import TacotronLite, tacotron_loss
from .text import TextTokenizer
from .utils import count_parameters, ensure_dir, format_param_count, get_device, load_config, set_seed


def build_model(config: dict[str, Any], tokenizer: TextTokenizer) -> TacotronLite:
    return TacotronLite(
        vocab_size=tokenizer.vocab_size,
        n_mels=int(config["audio"]["n_mels"]),
        config=config["model"],
        pad_id=tokenizer.pad_id,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    config: dict[str, Any],
    tokenizer: TextTokenizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
            "tokenizer": tokenizer.to_dict(),
            "model_type": "tacotron_lite",
        },
        path,
    )


@torch.no_grad()
def validate(model: TacotronLite, loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    mel_losses: list[float] = []
    post_losses: list[float] = []
    stop_losses: list[float] = []
    guide_losses: list[float] = []
    train_cfg = config["training"]
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["tokens"], batch["token_lens"], batch["mels"])
        loss, metrics = tacotron_loss(
            output,
            batch["mels"],
            batch["mel_lens"],
            batch["token_lens"],
            float(train_cfg.get("mel_loss_weight", 1.0)),
            float(train_cfg.get("stop_loss_weight", 0.5)),
            float(train_cfg.get("guided_attention_weight", 0.2)),
            float(train_cfg.get("guided_attention_sigma", 0.4)),
        )
        losses.append(float(loss.cpu()))
        mel_losses.append(metrics["mel_loss"])
        post_losses.append(metrics["postnet_mel_loss"])
        stop_losses.append(metrics["stop_loss"])
        guide_losses.append(metrics["guided_attention_loss"])
    model.train()
    if not losses:
        return {"val_loss": float("nan")}
    return {
        "val_loss": sum(losses) / len(losses),
        "val_mel_loss": sum(mel_losses) / len(mel_losses),
        "val_postnet_mel_loss": sum(post_losses) / len(post_losses),
        "val_stop_loss": sum(stop_losses) / len(stop_losses),
        "val_guided_attention_loss": sum(guide_losses) / len(guide_losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BananaTTS Tacotron-lite acoustic model.")
    parser.add_argument("--config", default="configs/bananatts_tacotron.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--percent", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 1337)))
    tokenizer = TextTokenizer.from_config(config["text"])
    cache_root = Path(config["dataset"]["cache_dir"])
    local_path = Path(args.local_path) if args.local_path else None
    if not (cache_root / "manifest.json").exists() or args.force_prepare:
        try:
            prepare_ljspeech(
                config,
                dataset_name=args.dataset,
                local_path=local_path,
                limit=args.limit,
                percent=args.percent,
                force=args.force_prepare,
            )
        except Exception:
            fallback = config["dataset"].get("fallback_name")
            if fallback and not local_path and fallback != (args.dataset or config["dataset"]["name"]):
                print(f"Primary dataset failed; retrying with fallback {fallback}")
                prepare_ljspeech(config, dataset_name=fallback, limit=args.limit, percent=args.percent, force=args.force_prepare)
            else:
                raise

    train_ds = BananaTTSDataset(cache_root, "train")
    val_ds = BananaTTSDataset(cache_root, "val")
    if train_ds.mel_stats:
        config.setdefault("audio", {})["mel_stats"] = train_ds.mel_stats
        print(
            "Using normalized mel targets:",
            f"mean={float(train_ds.mel_stats['mean']):.4f}",
            f"std={float(train_ds.mel_stats['std']):.4f}",
        )
    else:
        print("Warning: cache has no mel_stats. Re-run prepare_ljspeech.py --force for more stable training.")

    train_cfg = config["training"]
    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = build_model(config, tokenizer).to(device)
    print(f"Tacotron acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    step = 0
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {resume_path} at step {step}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(ensure_dir(train_cfg.get("runs_dir", "runs")) / "tacotron"))
    except Exception as exc:
        print(f"TensorBoard disabled: {exc}")

    ckpt_dir = ensure_dir(train_cfg.get("checkpoints_dir", "checkpoints_tacotron"))
    model.train()
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        pbar = tqdm(train_loader, desc=f"tacotron epoch {epoch + 1}")
        for batch in pbar:
            step += 1
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["tokens"], batch["token_lens"], batch["mels"])
                loss, metrics = tacotron_loss(
                    output,
                    batch["mels"],
                    batch["mel_lens"],
                    batch["token_lens"],
                    float(train_cfg.get("mel_loss_weight", 1.0)),
                    float(train_cfg.get("stop_loss_weight", 0.5)),
                    float(train_cfg.get("guided_attention_weight", 0.2)),
                    float(train_cfg.get("guided_attention_sigma", 0.4)),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                mel=f"{metrics['mel_loss']:.4f}",
                post=f"{metrics['postnet_mel_loss']:.4f}",
                attn=f"{metrics['guided_attention_loss']:.4f}",
            )

            if writer and step % int(train_cfg.get("log_interval", 20)) == 0:
                writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)

            if step % int(train_cfg.get("val_interval", 500)) == 0 and len(val_ds) > 0:
                val_metrics = validate(model, val_loader, device, config)
                print(f"step {step} validation: {val_metrics}")
                if writer:
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{key}", value, step)

            if step % int(train_cfg.get("save_interval", 1000)) == 0:
                save_checkpoint(ckpt_dir / "tacotron_latest.pt", model, optimizer, scaler, step, epoch, config, tokenizer)

        save_checkpoint(ckpt_dir / "tacotron_latest.pt", model, optimizer, scaler, step, epoch + 1, config, tokenizer)
    save_checkpoint(ckpt_dir / "tacotron_final.pt", model, optimizer, scaler, step, int(train_cfg["epochs"]), config, tokenizer)
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
