from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a BananaTTS Tacotron .pt checkpoint to safetensors.")
    parser.add_argument("--checkpoint", default="checkpoints_tacotron/tacotron_latest.pt")
    parser.add_argument("--out", default="checkpoints_tacotron/tacotron_model.safetensors")
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint has no 'model' key: {checkpoint_path}")

    state_dict = ckpt["model"]
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    if len(tensors) != len(state_dict):
        skipped = sorted(set(state_dict) - set(tensors))
        raise TypeError(f"Non-tensor model entries cannot be saved to safetensors: {skipped}")

    metadata = {
        "format": "pt",
        "model_type": str(ckpt.get("model_type", "tacotron_lite")),
        "source_checkpoint": checkpoint_path.name,
        "step": str(ckpt.get("step", "")),
        "epoch": str(ckpt.get("epoch", "")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_path, metadata=metadata)

    metadata_out = Path(args.metadata_out) if args.metadata_out else out_path.with_suffix(".json")
    if not metadata_out.is_absolute():
        metadata_out = ROOT / metadata_out
    sidecar = {
        "model_type": ckpt.get("model_type", "tacotron_lite"),
        "step": ckpt.get("step"),
        "epoch": ckpt.get("epoch"),
        "config": json_safe(ckpt.get("config", {})),
        "tokenizer": json_safe(ckpt.get("tokenizer", {})),
        "weights": out_path.name,
    }
    with metadata_out.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved weights: {out_path}")
    print(f"Saved metadata: {metadata_out}")
    print(f"Tensors: {len(tensors)}")
    print(f"Epoch: {ckpt.get('epoch')} Step: {ckpt.get('step')}")


if __name__ == "__main__":
    main()
