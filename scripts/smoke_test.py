from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bananatts.audio import save_wav
from bananatts.data import BananaTTSDataset, collate_batch, make_synthetic_batch, prepare_ljspeech
from bananatts.models.acoustic import FastSpeech2AcousticModel, acoustic_loss
from bananatts.models.vocoder import GriffinLimVocoder
from bananatts.text import TextTokenizer
from bananatts.utils import count_parameters, ensure_dir, format_param_count, get_device, load_config, set_seed


def main() -> None:
    config = load_config(ROOT / "configs/bananatts_20m.yaml")
    set_seed(int(config["project"].get("seed", 1337)))
    tokenizer = TextTokenizer.from_config(config["text"])
    cache_root = ROOT / config["dataset"]["cache_dir"]
    batch_items = None

    try:
        prepare_ljspeech(config, dataset_name=config["dataset"]["name"], limit=10)
        dataset = BananaTTSDataset(cache_root, "train")
        batch_items = [dataset[i] for i in range(min(2, len(dataset)))]
        print(f"Smoke test loaded {len(dataset)} prepared training samples from LJSpeech cache.")
    except Exception as exc:
        print(f"LJSpeech mini-prepare failed, using synthetic batch for shape smoke test: {exc}")
        batch_items = make_synthetic_batch(tokenizer, config["audio"], batch_size=2)

    batch = collate_batch(batch_items, pad_id=tokenizer.pad_id)
    device = get_device()
    model = FastSpeech2AcousticModel(
        tokenizer.vocab_size,
        int(config["audio"]["n_mels"]),
        config["model"],
        tokenizer.pad_id,
        special_token_ids=(tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id),
    ).to(device)
    print(f"Acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
    model.train()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(batch["tokens"], batch["token_lens"], durations=batch["durations"], max_frames=batch["mels"].shape[1])
    loss, metrics = acoustic_loss(output, batch["mels"], batch["durations"], batch["mel_lens"], batch["token_lens"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"Forward/training step OK: loss={float(loss.detach().cpu()):.4f}, metrics={metrics}")

    model.eval()
    with torch.no_grad():
        infer = model(batch["tokens"][:1], batch["token_lens"][:1], durations=None)
    mel = infer.mel[0, : infer.regulated_lengths[0]].detach().cpu()
    wav = GriffinLimVocoder(config["audio"])(mel)
    out_path = ensure_dir(ROOT / "samples") / "smoke_test.wav"
    save_wav(out_path, wav, int(config["audio"]["sample_rate"]))
    print(f"Saved smoke audio: {out_path}")


if __name__ == "__main__":
    main()
