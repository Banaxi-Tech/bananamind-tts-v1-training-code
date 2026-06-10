from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio import MelSpectrogram, resample_linear, to_mono, validate_audio
from .text import TextTokenizer
from .utils import load_json, save_config, save_json


@dataclass
class PreparedSample:
    path: str
    text: str
    normalized_text: str
    frames: int
    tokens: int


def uniform_durations(num_tokens: int, num_frames: int) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    base = num_frames // num_tokens
    rem = num_frames % num_tokens
    durations = torch.full((num_tokens,), base, dtype=torch.long)
    if rem:
        durations[:rem] += 1
    durations = durations.clamp_min(1)
    diff = int(durations.sum().item() - num_frames)
    if diff > 0:
        for i in range(num_tokens - 1, -1, -1):
            take = min(diff, max(0, int(durations[i].item()) - 1))
            durations[i] -= take
            diff -= take
            if diff == 0:
                break
    return durations


def uniform_content_durations(
    token_ids: list[int],
    num_frames: int,
    silent_token_ids: set[int],
) -> torch.Tensor:
    """Assign mel frames only to content tokens.

    BOS/EOS/PAD are sequence markers, not speech sounds. Giving them duration
    trains the model to synthesize audio for special tokens, which is especially
    destructive for short prompts such as "Hello".
    """
    content_indices = [i for i, token_id in enumerate(token_ids) if token_id not in silent_token_ids]
    durations = torch.zeros(len(token_ids), dtype=torch.long)
    if not content_indices:
        return uniform_durations(len(token_ids), num_frames)
    content_durations = uniform_durations(len(content_indices), num_frames)
    for idx, duration in zip(content_indices, content_durations, strict=True):
        durations[idx] = duration
    return durations


def _record_text(record: dict[str, Any]) -> str:
    for key in ("normalized_text", "text", "transcription", "sentence"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"No usable text field found. Available keys: {sorted(record.keys())}")


def _record_audio(record: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio = record.get("audio") or record.get("file") or record.get("audio_file")
    if isinstance(audio, dict):
        array = audio.get("array")
        sr = audio.get("sampling_rate")
        if array is not None and sr is not None:
            return np.asarray(array, dtype=np.float32), int(sr)
        path = audio.get("path")
        if path:
            return _load_audio_path(path)
    if isinstance(audio, (str, Path)):
        return _load_audio_path(audio)
    raise KeyError(f"No usable audio field found. Available keys: {sorted(record.keys())}")


def _load_audio_path(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(sr)
    except ImportError:
        pass
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        return wav.squeeze(0).numpy(), int(sr)
    except ImportError as exc:
        try:
            return _load_wav_stdlib(path)
        except Exception as wav_exc:
            raise RuntimeError("Loading audio paths requires soundfile, torchaudio, or a PCM WAV readable by wave") from wav_exc


def _load_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, sample_rate


def load_hf_ljspeech(dataset_name: str, split: str, cache_dir: str | None = None):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load LJSpeech from Hugging Face") from exc

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(decode=True))
        except Exception as exc:
            print(f"Warning: Hugging Face Audio(decode=True) failed ({exc}); retrying with decode=False")
            ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def load_local_ljspeech(local_path: str | Path) -> list[dict[str, Any]]:
    root = Path(local_path)
    metadata_paths = [root / "metadata.csv"]
    if not metadata_paths[0].exists():
        metadata_paths = [
            path
            for path in (
                root / "metadata_train.csv",
                root / "metadata_dev.csv",
                root / "metadata_test.csv",
            )
            if path.exists()
        ]
    if not metadata_paths:
        metadata_paths = sorted(root.glob("metadata*.csv"))
    wavs_dir = root / "wavs"
    if not metadata_paths:
        raise FileNotFoundError(f"Missing LJSpeech-style metadata CSV in {root}")
    if not wavs_dir.exists() or not wavs_dir.is_dir():
        raise FileNotFoundError(f"Missing LJSpeech wavs directory: {wavs_dir}")
    wav_count = sum(1 for _ in wavs_dir.glob("*.wav"))
    if wav_count == 0:
        raise FileNotFoundError(f"No WAV files found in {wavs_dir}")

    rows: list[tuple[str, str]] = []
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    print(
                        f"Warning: skipping malformed metadata row {metadata_path.name}:{line_no}: "
                        "expected wav_id|transcript|normalized_transcript"
                    )
                    continue
                wav_id = parts[0].strip()
                transcript = parts[1].strip()
                normalized = parts[2].strip() if len(parts) > 2 else ""
                text = normalized or transcript
                rows.append((wav_id, text))

    records: list[dict[str, Any]] = []
    missing = 0
    missing_warning_limit = 20
    for wav_id, text in rows:
        wav_path = wavs_dir / f"{wav_id}.wav"
        if not wav_path.exists():
            missing += 1
            if missing <= missing_warning_limit:
                print(f"Warning: missing WAV for metadata row {wav_id}: {wav_path}")
            continue
        records.append({"text": text, "audio": str(wav_path), "wav_id": wav_id})

    print("metadata files:")
    for metadata_path in metadata_paths:
        print(f"  {metadata_path.name}")
    print(f"total rows: {len(rows)}")
    print(f"valid rows: {len(records)}")
    if missing:
        if missing > missing_warning_limit:
            print(f"Warning: suppressed {missing - missing_warning_limit} additional missing WAV warnings")
        print(f"missing WAV rows skipped: {missing}")
    if not records:
        raise RuntimeError(f"No metadata rows in {metadata_path} had matching WAV files")
    print("first 3 examples:")
    for record in records[:3]:
        print(f"  {record['wav_id']}: {record['text']}")
    return records


def select_records(records: Any, total: int, limit: int | None, percent: float | None, default_limit: Any = None) -> tuple[Any, int]:
    selected_count = total
    if percent is not None:
        selected_count = max(1, int(total * float(percent) / 100.0))
    if limit is not None:
        selected_count = min(selected_count, int(limit))
    elif default_limit:
        selected_count = min(selected_count, int(default_limit))
    selected_count = min(selected_count, total)
    if hasattr(records, "select"):
        return records.select(range(selected_count)), selected_count
    return records[:selected_count], selected_count


def prepare_ljspeech(
    config: dict[str, Any],
    dataset_name: str | None = None,
    local_path: str | Path | None = None,
    limit: int | None = None,
    percent: float | None = None,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]
    text_cfg = config["text"]
    dataset_name = dataset_name or dataset_cfg["name"]
    if local_path is None and dataset_cfg.get("local_path"):
        local_path = dataset_cfg["local_path"]
    split = dataset_cfg.get("split", "train")
    cache_root = Path(cache_root or dataset_cfg["cache_dir"])
    samples_dir = cache_root / "samples"
    manifest_path = cache_root / "manifest.json"
    split_path = cache_root / "split.json"

    if manifest_path.exists() and split_path.exists() and not force:
        manifest = load_json(manifest_path)
        print(f"Using cached dataset at {cache_root} ({len(manifest['samples'])} samples)")
        return manifest

    tokenizer = TextTokenizer.from_config(text_cfg)
    mel_fn = MelSpectrogram(audio_cfg)
    source = str(local_path) if local_path else dataset_name
    ds = load_local_ljspeech(local_path) if local_path else load_hf_ljspeech(dataset_name, split=split)
    total = len(ds)
    ds, selected_count = select_records(ds, total, limit, percent, dataset_cfg.get("limit"))
    print(f"selected rows: {selected_count}")

    samples_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, str]] = []
    prepared: list[PreparedSample] = []
    skipped: list[dict[str, str]] = []
    mel_sum = 0.0
    mel_sumsq = 0.0
    mel_count = 0
    mel_min = float("inf")
    mel_max = float("-inf")
    min_seconds = float(dataset_cfg.get("min_audio_seconds", 0.25))
    max_seconds = float(dataset_cfg.get("max_audio_seconds", 20.0))
    sample_rate = int(audio_cfg["sample_rate"])
    max_tokens = int(text_cfg.get("max_tokens", 256))
    silent_token_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}

    for idx, record in enumerate(tqdm(ds, desc="Preparing LJSpeech")):
        try:
            raw_text = _record_text(record)
            normalized = tokenizer.normalize(raw_text)
            token_ids = tokenizer.encode(normalized)
            if len(token_ids) > max_tokens:
                raise ValueError(f"text too long: {len(token_ids)} tokens")
            audio, sr = _record_audio(record)
            wav = to_mono(audio)
            wav = resample_linear(wav, sr, sample_rate)
            validate_audio(wav, sample_rate, min_seconds, max_seconds)
            mel = mel_fn(wav)
            if mel.shape[0] < 2:
                raise ValueError("mel has fewer than 2 frames")
            mel_float = mel.to(torch.float32)
            mel_sum += float(mel_float.sum().item())
            mel_sumsq += float(mel_float.square().sum().item())
            mel_count += int(mel_float.numel())
            mel_min = min(mel_min, float(mel_float.min().item()))
            mel_max = max(mel_max, float(mel_float.max().item()))
            durations = uniform_content_durations(token_ids, mel.shape[0], silent_token_ids)
            out_path = samples_dir / f"{idx:06d}.pt"
            torch.save(
                {
                    "tokens": torch.tensor(token_ids, dtype=torch.long),
                    "durations": durations,
                    "mel": mel_float,
                    "wav": wav.to(torch.float32),
                    "text": raw_text,
                    "normalized_text": normalized,
                },
                out_path,
            )
            rel = out_path.relative_to(cache_root).as_posix()
            prepared.append(PreparedSample(rel, raw_text, normalized, int(mel.shape[0]), len(token_ids)))
            if len(examples) < 3:
                examples.append({"text": raw_text, "normalized": normalized})
        except Exception as exc:
            skipped.append({"index": str(idx), "error": str(exc)})

    if not prepared:
        raise RuntimeError("No LJSpeech samples were prepared successfully")
    mel_mean = mel_sum / max(1, mel_count)
    mel_var = max(1e-8, mel_sumsq / max(1, mel_count) - mel_mean * mel_mean)
    mel_std = math.sqrt(mel_var)

    val_ratio = float(dataset_cfg.get("validation_ratio", 0.02))
    val_count = max(1, int(math.ceil(len(prepared) * val_ratio))) if len(prepared) > 1 else 0
    indices = list(range(len(prepared)))
    train_indices = indices[:-val_count] if val_count else indices
    val_indices = indices[-val_count:] if val_count else []

    manifest = {
        "dataset": source,
        "source_type": "local" if local_path else "huggingface",
        "sample_rate": sample_rate,
        "audio": audio_cfg,
        "text": tokenizer.to_dict(),
        "duration_target": "uniform_content_v1",
        "contains_wav": True,
        "mel_stats": {
            "mean": mel_mean,
            "std": mel_std,
            "min": mel_min,
            "max": mel_max,
            "count": mel_count,
            "normalized_training": True,
        },
        "samples": [sample.__dict__ for sample in prepared],
        "examples": examples,
        "skipped": skipped[:50],
    }
    split_data = {"train": train_indices, "val": val_indices}
    save_json(manifest, manifest_path)
    save_json(split_data, split_path)
    save_config({"audio": audio_cfg, "text": tokenizer.to_dict()}, cache_root / "feature_config.yaml")
    print(f"Prepared {len(prepared)} samples in {cache_root}; skipped {len(skipped)}")
    print(f"Train/val split: {len(train_indices)}/{len(val_indices)}")
    print(f"Mel stats: mean={mel_mean:.4f}, std={mel_std:.4f}, min={mel_min:.4f}, max={mel_max:.4f}")
    for ex in examples:
        print(f"Example: {ex['normalized']}")
    return manifest


class BananaTTSDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache_root: str | Path, split: str = "train"):
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        split_path = self.cache_root / "split.json"
        if not manifest_path.exists() or not split_path.exists():
            raise FileNotFoundError(f"Missing prepared dataset in {self.cache_root}; run scripts/prepare_ljspeech.py")
        self.manifest = load_json(manifest_path)
        self.split_data = load_json(split_path)
        self.indices = list(self.split_data[split])
        self.samples = self.manifest["samples"]
        self.mel_stats = self.manifest.get("mel_stats")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[self.indices[idx]]
        data = torch.load(self.cache_root / sample["path"], map_location="cpu")
        if self.mel_stats and self.mel_stats.get("normalized_training", False):
            mean = float(self.mel_stats["mean"])
            std = max(float(self.mel_stats["std"]), 1e-5)
            data["mel"] = (data["mel"] - mean) / std
        return data


def collate_batch(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    token_lens = torch.tensor([item["tokens"].numel() for item in batch], dtype=torch.long)
    mel_lens = torch.tensor([item["mel"].shape[0] for item in batch], dtype=torch.long)
    max_tokens = int(token_lens.max().item())
    max_mels = int(mel_lens.max().item())
    n_mels = int(batch[0]["mel"].shape[1])
    tokens = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
    durations = torch.zeros((len(batch), max_tokens), dtype=torch.long)
    mels = torch.zeros((len(batch), max_mels, n_mels), dtype=torch.float32)
    texts: list[str] = []
    for i, item in enumerate(batch):
        t_len = item["tokens"].numel()
        m_len = item["mel"].shape[0]
        tokens[i, :t_len] = item["tokens"]
        durations[i, :t_len] = item["durations"]
        mels[i, :m_len] = item["mel"]
        texts.append(item.get("normalized_text", item.get("text", "")))
    return {
        "tokens": tokens,
        "token_lens": token_lens,
        "durations": durations,
        "mels": mels,
        "mel_lens": mel_lens,
        "texts": texts,
    }


def make_synthetic_batch(tokenizer: TextTokenizer, audio_cfg: dict[str, Any], batch_size: int = 2) -> list[dict[str, Any]]:
    texts = [
        "hello from banana tts.",
        "this is a small smoke test.",
        "text to speech starts with clean shapes.",
    ]
    n_mels = int(audio_cfg["n_mels"])
    batch = []
    for i in range(batch_size):
        token_ids = tokenizer.encode(texts[i % len(texts)])
        tokens = torch.tensor(token_ids, dtype=torch.long)
        frames = max((tokens.numel() - 2) * 4, 32)
        mel = torch.randn(frames, n_mels).mul(0.25).sub(4.0)
        durations = uniform_content_durations(token_ids, frames, {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id})
        batch.append({"tokens": tokens, "durations": durations, "mel": mel, "text": texts[i % len(texts)]})
    return batch
