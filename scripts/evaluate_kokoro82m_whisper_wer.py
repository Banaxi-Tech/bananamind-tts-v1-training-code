from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bananatts.audio import save_wav
from bananatts.kokoro_pytorch import KModel
from evaluate_libritts_wer import (
    DEFAULT_ARCHIVE,
    build_entries,
    compute_wer_fields,
    read_jsonl_rows,
    select_samples,
    source_duration_seconds,
    wav_seconds,
    write_jsonl,
)


DEFAULT_KOKORO_DIR = Path("/home/banaxi/ai-models/kokoro-82m")
DEFAULT_WHISPER_DIR = Path("/home/banaxi/ai-models/whisper-large-v3")
DEFAULT_OUT_DIR = ROOT / "outputs" / "kokoro82m_whisper_large_v3_wer"
KOKORO_SAMPLE_RATE = 24000


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def clean_phonemes(phonemes: str, vocab: dict[str, int]) -> tuple[str, list[str]]:
    phonemes = " ".join(phonemes.split())
    unknown = sorted({char for char in phonemes if char not in vocab})
    cleaned = "".join(char for char in phonemes if char in vocab)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, unknown


def phonemize_espeak(text: str, voice: str, vocab: dict[str, int]) -> tuple[str, list[str]]:
    if shutil.which("espeak-ng") is None:
        raise RuntimeError("Missing espeak-ng. Install it first, then rerun this script.")
    proc = subprocess.run(
        ["espeak-ng", "-q", "-v", voice, "--ipa", text],
        check=True,
        capture_output=True,
        text=True,
    )
    return clean_phonemes(proc.stdout, vocab)


def split_phonemes(phonemes: str, max_chars: int) -> list[str]:
    words = phonemes.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(word) > max_chars:
            chunks.append(word[:max_chars])
            word = word[max_chars:]
        current = word
    if current:
        chunks.append(current)
    return chunks


class KokoroSynthesizer:
    def __init__(self, kokoro_dir: Path, voice_name: str, device: str | None, espeak_voice: str, speed: float, max_phoneme_chars: int):
        self.kokoro_dir = kokoro_dir
        self.voice_name = voice_name
        self.espeak_voice = espeak_voice
        self.speed = speed
        self.max_phoneme_chars = max_phoneme_chars
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = KModel(kokoro_dir / "config.json", kokoro_dir / "kokoro-v1_0.pth").to(self.device).eval()
        self.voice_pack = torch.load(kokoro_dir / "voices" / f"{voice_name}.pt", map_location="cpu", weights_only=True)

    @torch.no_grad()
    def synthesize(self, text: str) -> tuple[torch.Tensor, str, list[str]]:
        phonemes, unknown = phonemize_espeak(text, self.espeak_voice, self.model.vocab)
        chunks = split_phonemes(phonemes, self.max_phoneme_chars)
        audio_chunks: list[torch.Tensor] = []
        for chunk in chunks:
            if not chunk:
                continue
            ref_s = self.voice_pack[len(chunk) - 1]
            audio = self.model(chunk, ref_s, speed=self.speed)
            audio_chunks.append(audio.float().cpu())
        if not audio_chunks:
            raise RuntimeError(f"No Kokoro audio chunks generated for text: {text!r}")
        pause = torch.zeros(int(KOKORO_SAMPLE_RATE * 0.08), dtype=torch.float32)
        audio = torch.cat([part for chunk in audio_chunks for part in (chunk, pause)])
        return audio, phonemes, unknown


def synthesize_missing(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    synth = KokoroSynthesizer(
        kokoro_dir=resolve_path(args.kokoro_dir),
        voice_name=args.voice,
        device=args.device,
        espeak_voice=args.espeak_voice,
        speed=args.speed,
        max_phoneme_chars=args.max_phoneme_chars,
    )
    print(f"Loaded Kokoro 82M on {synth.device}; voice={args.voice}; torch.load weights_only=True")
    for row in tqdm(entries, desc="Synthesizing Kokoro"):
        wav_path = Path(row["generated_wav"])
        if wav_path.exists() and not args.force_synthesis:
            row["generated_seconds"] = wav_seconds(wav_path)
            row["tts_backend"] = "kokoro-82m-pytorch-espeak"
            row.pop("synthesis_error", None)
            continue
        try:
            wav, phonemes, unknown = synth.synthesize(str(row["reference_text"]))
            save_wav(wav_path, wav, KOKORO_SAMPLE_RATE)
            row["generated_seconds"] = wav.numel() / float(KOKORO_SAMPLE_RATE)
            row["phonemes"] = phonemes
            row["unknown_phoneme_chars"] = unknown
            row["sample_rate"] = KOKORO_SAMPLE_RATE
            row["tts_backend"] = "kokoro-82m-pytorch-espeak"
            row.pop("synthesis_error", None)
        except Exception as exc:  # noqa: BLE001 - keep the long eval resumable.
            row["synthesis_error"] = repr(exc)
            if args.stop_on_error:
                raise


def read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}: {path}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, audio.shape[0] - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def load_whisper_pipeline(args: argparse.Namespace):
    from transformers import pipeline

    model_path = resolve_path(args.whisper_model)
    device = torch.device(args.whisper_device if args.whisper_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe_device = 0 if device.type == "cuda" else -1
    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "dtype": torch_dtype,
        "device": pipe_device,
        "batch_size": args.whisper_batch_size,
    }
    if args.whisper_chunk_length > 0:
        kwargs["chunk_length_s"] = args.whisper_chunk_length
    return pipeline("automatic-speech-recognition", **kwargs)


def transcribe_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    pipe = load_whisper_pipeline(args)
    generate_kwargs = {"language": args.language, "task": "transcribe"} if args.language else {"task": "transcribe"}
    for row in tqdm(entries, desc="Transcribing with local Whisper"):
        wav_path = Path(row["generated_wav"])
        if not wav_path.exists() or row.get("synthesis_error"):
            continue
        if row.get("transcript") and not args.force_transcribe:
            compute_wer_fields(row)
            continue
        try:
            audio, sample_rate = read_wav_float32(wav_path)
            whisper_rate = int(getattr(pipe.feature_extractor, "sampling_rate", 16000))
            audio = resample_linear(audio, sample_rate, whisper_rate)
            result = pipe({"array": audio, "sampling_rate": whisper_rate}, generate_kwargs=generate_kwargs)
            transcript = result["text"] if isinstance(result, dict) else str(result)
            row["transcript"] = transcript.strip()
            row["whisper_response"] = result
            row["transcription_backend"] = "local-whisper-large-v3"
            row.pop("transcription_error", None)
            compute_wer_fields(row)
        except Exception as exc:  # noqa: BLE001 - keep partial eval results.
            row["transcription_error"] = repr(exc)
            if args.stop_on_error:
                raise


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in entries if Path(row["generated_wav"]).exists() and not row.get("synthesis_error")]
    generated_seconds = sum(float(row.get("generated_seconds") or 0.0) for row in generated)
    transcribed = [row for row in generated if "transcript" in row and row.get("wer_reference_words")]
    ref_words = sum(int(row.get("wer_reference_words") or 0) for row in transcribed)
    edits = sum(int(row.get("wer_edits") or 0) for row in transcribed)
    return {
        "evaluation_target": "Kokoro 82M generated TTS audio",
        "reference_text_source": "LibriTTS test-clean .normalized.txt",
        "transcription_model": "local openai/whisper-large-v3",
        "wer_normalizer": "lowercase_punctuation_apostrophe_digits_to_words",
        "requested_examples": len(entries),
        "synthesized_examples": len(generated),
        "synthesis_failures": sum(1 for row in entries if row.get("synthesis_error")),
        "generated_seconds": generated_seconds,
        "generated_minutes": generated_seconds / 60.0,
        "generated_hours": generated_seconds / 3600.0,
        "model_tts_generated_seconds": generated_seconds,
        "model_tts_generated_minutes": generated_seconds / 60.0,
        "model_tts_generated_hours": generated_seconds / 3600.0,
        "transcribed_examples": len(transcribed),
        "transcription_failures": sum(1 for row in entries if row.get("transcription_error")),
        "wer_reference_words": ref_words,
        "wer_edits": edits,
        "wer": edits / ref_words if ref_words else None,
    }


def recompute_existing_wer(manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    entries = read_jsonl_rows(manifest_path)
    if not entries:
        raise SystemExit(f"No rows found in {manifest_path}")
    for row in entries:
        if row.get("transcript"):
            compute_wer_fields(row)
    write_jsonl(manifest_path, entries)
    summary = summarize(entries)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize LibriTTS test-clean text with local PyTorch Kokoro 82M, then transcribe with local Whisper Large V3 and compute WER."
    )
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="Path to LibriTTS test-clean tar.gz.")
    parser.add_argument("--limit", type=int, default=1850, help="Number of examples to select from the archive.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated audio, manifests, and summaries.")
    parser.add_argument("--kokoro-dir", default=str(DEFAULT_KOKORO_DIR), help="Directory containing config.json, kokoro-v1_0.pth, and voices/*.pt.")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice file name without .pt.")
    parser.add_argument("--espeak-voice", default="en-us", help="espeak-ng voice used for IPA phonemization.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-phoneme-chars", type=int, default=480)
    parser.add_argument("--device", default=None, help="Kokoro device override, e.g. cuda or cpu.")
    parser.add_argument("--force-synthesis", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Only select examples and report source-audio hours from the archive.")
    parser.add_argument("--transcribe", action="store_true", help="Use local Whisper Large V3 after synthesis.")
    parser.add_argument("--whisper-model", default=str(DEFAULT_WHISPER_DIR), help="Local Whisper Large V3 model directory.")
    parser.add_argument("--whisper-device", default=None, help="Whisper device override, e.g. cuda or cpu.")
    parser.add_argument("--whisper-batch-size", type=int, default=8)
    parser.add_argument("--whisper-chunk-length", type=float, default=0.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--force-transcribe", action="store_true")
    parser.add_argument("--recompute-wer-only", action="store_true", help="Recompute WER from existing saved transcripts without synthesis or transcription.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    archive_path = resolve_path(args.archive)
    out_dir = resolve_path(args.out_dir)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    samples_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "summary.json"

    if args.recompute_wer_only:
        summary = recompute_existing_wer(manifest_path, summary_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if not archive_path.exists():
        raise SystemExit(f"Missing archive: {archive_path}")

    samples = select_samples(archive_path, args.limit, include_source_duration=args.metadata_only)
    write_jsonl(samples_path, [sample.__dict__ for sample in samples])
    entries = build_entries(samples, manifest_path, audio_dir)

    if args.metadata_only:
        source_seconds = source_duration_seconds(archive_path, samples)
        summary = {
            "requested_examples": len(samples),
            "source_seconds": source_seconds,
            "source_minutes": source_seconds / 60.0,
            "source_hours": source_seconds / 3600.0,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    synthesize_missing(entries, args)
    write_jsonl(manifest_path, entries)

    if args.transcribe:
        transcribe_entries(entries, args)
        write_jsonl(manifest_path, entries)

    summary = summarize(entries)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.transcribe:
        print("No Whisper transcription was run. Add --transcribe to calculate WER with local Whisper Large V3.")


if __name__ == "__main__":
    main()
