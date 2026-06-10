from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tarfile
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bananatts.audio import save_wav
from bananatts.models.vocoder import load_vocoder_from_config
from bananatts.synthesize_tacotron import load_model, synthesize
from bananatts.text import TextTokenizer
from bananatts.utils import get_device, load_config


DEFAULT_ARCHIVE = Path("/home/banaxi/Downloads/test-clean.tar.gz")
DEFAULT_OUT_DIR = ROOT / "outputs" / "bananamind_v2v3_libritts_wer"
DEFAULT_STT_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "mistralai/voxtral-mini-transcribe"


@dataclass(frozen=True)
class LibriTTSSample:
    index: int
    sample_id: str
    text: str
    wav_member: str
    text_member: str
    source_seconds: float | None = None


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def normalized_member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def sample_id_from_text_member(name: str) -> str:
    return Path(name).name.replace(".normalized.txt", "")


def corresponding_wav_member(text_member: str) -> str:
    return text_member.replace(".normalized.txt", ".wav")


def wav_duration_from_bytes(data: bytes) -> float:
    with wave.open(BytesIO(data), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def select_samples(archive_path: Path, limit: int, include_source_duration: bool = False) -> list[LibriTTSSample]:
    samples: list[LibriTTSSample] = []
    pending: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized_name = normalized_member_name(member.name)
            if normalized_name.endswith(".normalized.txt"):
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                text = file_obj.read().decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                base = normalized_name.replace(".normalized.txt", "")
                item = pending.setdefault(base, {})
                item["text"] = text
                item["text_member"] = member.name
            elif normalized_name.endswith(".wav"):
                base = normalized_name[:-4]
                item = pending.setdefault(base, {})
                item["wav_member"] = member.name
                if include_source_duration:
                    file_obj = archive.extractfile(member)
                    if file_obj is not None:
                        item["source_seconds"] = wav_duration_from_bytes(file_obj.read())
            else:
                continue

            item = pending.get(normalized_name.replace(".normalized.txt", "").replace(".wav", ""))
            if not item or "text" not in item or "wav_member" not in item:
                continue
            samples.append(
                LibriTTSSample(
                    index=len(samples),
                    sample_id=Path(normalized_name.replace(".normalized.txt", "").replace(".wav", "")).name,
                    text=str(item["text"]),
                    wav_member=str(item["wav_member"]),
                    text_member=str(item["text_member"]),
                    source_seconds=item.get("source_seconds"),
                )
            )
            pending.pop(normalized_name.replace(".normalized.txt", "").replace(".wav", ""), None)
            if len(samples) >= limit:
                break
    if len(samples) < limit:
        print(f"Warning: requested {limit} examples but only found {len(samples)} usable examples.")
    return samples


def source_duration_seconds(archive_path: Path, samples: list[LibriTTSSample]) -> float:
    if all(sample.source_seconds is not None for sample in samples):
        return sum(float(sample.source_seconds) for sample in samples)
    total = 0.0
    wanted = {normalized_member_name(sample.wav_member) for sample in samples}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in tqdm(archive, desc="Reading source durations"):
            if not member.isfile() or normalized_member_name(member.name) not in wanted:
                continue
            file_obj = archive.extractfile(member)
            if file_obj is not None:
                total += wav_duration_from_bytes(file_obj.read())
    return total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl_rows(path)
    return {row["id"]: row for row in rows if isinstance(row.get("id"), str)}


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
ORDINAL_ONES = {
    0: "zeroth",
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
ORDINAL_TENS = {
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}
NUMBER_RE = re.compile(r"(?<![a-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?![a-z0-9])")
ORDINAL_RE = re.compile(r"(?<![a-z0-9])(\d+)(st|nd|rd|th)(?![a-z0-9])")


def int_to_cardinal_words(value: int) -> str:
    if value < 0:
        return "minus " + int_to_cardinal_words(abs(value))
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = value // 10 * 10
        ones = value % 10
        return TENS[tens] if ones == 0 else f"{TENS[tens]} {ONES[ones]}"
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        return f"{ONES[hundreds]} hundred" if rest == 0 else f"{ONES[hundreds]} hundred {int_to_cardinal_words(rest)}"
    if value == 1000:
        return "one thousand"
    if 1000 <= value <= 2099:
        first = value // 100
        last = value % 100
        if value == 2000:
            return "two thousand"
        if 2001 <= value <= 2009:
            return f"two thousand {ONES[last]}"
        if last == 0:
            return f"{int_to_cardinal_words(first)} hundred"
        if last < 10:
            return f"{int_to_cardinal_words(first)} oh {ONES[last]}"
        return f"{int_to_cardinal_words(first)} {int_to_cardinal_words(last)}"
    if value < 1_000_000:
        thousands = value // 1000
        rest = value % 1000
        return (
            f"{int_to_cardinal_words(thousands)} thousand"
            if rest == 0
            else f"{int_to_cardinal_words(thousands)} thousand {int_to_cardinal_words(rest)}"
        )
    if value < 1_000_000_000:
        millions = value // 1_000_000
        rest = value % 1_000_000
        return (
            f"{int_to_cardinal_words(millions)} million"
            if rest == 0
            else f"{int_to_cardinal_words(millions)} million {int_to_cardinal_words(rest)}"
        )
    return " ".join(ONES[int(char)] for char in str(value))


def int_to_ordinal_words(value: int) -> str:
    if value < 20:
        return ORDINAL_ONES[value]
    if value < 100:
        tens = value // 10 * 10
        ones = value % 10
        return ORDINAL_TENS[tens] if ones == 0 else f"{TENS[tens]} {ORDINAL_ONES[ones]}"
    cardinal = int_to_cardinal_words(value)
    words = cardinal.split()
    words[-1] = int_to_ordinal_words(int(words[-1])) if words[-1].isdigit() else {
        "one": "first",
        "two": "second",
        "three": "third",
        "four": "fourth",
        "five": "fifth",
        "six": "sixth",
        "seven": "seventh",
        "eight": "eighth",
        "nine": "ninth",
        "ten": "tenth",
        "eleven": "eleventh",
        "twelve": "twelfth",
        "thirteen": "thirteenth",
        "fourteen": "fourteenth",
        "fifteen": "fifteenth",
        "sixteen": "sixteenth",
        "seventeen": "seventeenth",
        "eighteen": "eighteenth",
        "nineteen": "nineteenth",
        "twenty": "twentieth",
        "thirty": "thirtieth",
        "forty": "fortieth",
        "fifty": "fiftieth",
        "sixty": "sixtieth",
        "seventy": "seventieth",
        "eighty": "eightieth",
        "ninety": "ninetieth",
        "hundred": "hundredth",
        "thousand": "thousandth",
        "million": "millionth",
    }.get(words[-1], words[-1])
    return " ".join(words)


def expand_numbers_for_wer(text: str) -> str:
    def ordinal_repl(match: re.Match[str]) -> str:
        return " " + int_to_ordinal_words(int(match.group(1).replace(",", ""))) + " "

    def cardinal_repl(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", "")
        if len(raw) > 1 and raw.startswith("0"):
            return " " + " ".join(ONES[int(char)] for char in raw) + " "
        return " " + int_to_cardinal_words(int(raw)) + " "

    text = ORDINAL_RE.sub(ordinal_repl, text)
    return NUMBER_RE.sub(cardinal_repl, text)


def normalize_for_wer(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("'", "")
    text = expand_numbers_for_wer(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            substitute = previous[j - 1] + (0 if ref_word == hyp_word else 1)
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            current.append(min(substitute, insert, delete))
        previous = current
    return previous[-1]


def transcribe_openrouter(
    audio_path: Path,
    api_key: str,
    model: str,
    endpoint: str,
    language: str,
    timeout: float,
    retries: int,
    http_referer: str | None,
    x_title: str | None,
) -> tuple[str, dict[str, Any]]:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
    payload: dict[str, Any] = {
        "model": model,
        "input_audio": {
            "data": audio_b64,
            "format": audio_path.suffix.lstrip(".").lower() or "wav",
        },
    }
    if language:
        payload["language"] = language

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            text = data.get("text")
            if not isinstance(text, str):
                raise ValueError(f"OpenRouter response did not contain text: {data}")
            return text.strip(), data
        except Exception as exc:  # noqa: BLE001 - report API and network failures with context.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"OpenRouter transcription failed for {audio_path}: {last_error}") from last_error


def build_entries(samples: list[LibriTTSSample], manifest_path: Path, audio_dir: Path) -> list[dict[str, Any]]:
    existing = read_jsonl_by_id(manifest_path)
    entries: list[dict[str, Any]] = []
    for sample in samples:
        wav_path = audio_dir / f"{sample.index:05d}_{sample.sample_id}.wav"
        row = dict(existing.get(sample.sample_id, {}))
        row.update(
            {
                "index": sample.index,
                "id": sample.sample_id,
                "reference_text": sample.text,
                "source_wav_member": sample.wav_member,
                "source_text_member": sample.text_member,
                "generated_wav": str(wav_path),
            }
        )
        entries.append(row)
    return entries


def synthesize_missing(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    config = load_config(resolve_path(args.config))
    tokenizer = TextTokenizer.from_config(config["text"])
    device = torch.device(args.device) if args.device else get_device()
    model = load_model(config, tokenizer, str(resolve_path(args.checkpoint)), device)
    vocoder_type = None if args.vocoder == "auto" else args.vocoder
    vocoder = load_vocoder_from_config(
        config,
        device,
        checkpoint=str(resolve_path(args.vocoder_checkpoint)) if args.vocoder_checkpoint else None,
        vocoder_type=vocoder_type,
    )
    sample_rate = int(config["audio"]["sample_rate"])

    for row in tqdm(entries, desc="Synthesizing TTS"):
        wav_path = Path(row["generated_wav"])
        if wav_path.exists() and not args.force_synthesis:
            row["generated_seconds"] = wav_seconds(wav_path)
            row.pop("synthesis_error", None)
            continue
        try:
            wav = synthesize(
                config,
                model,
                tokenizer,
                str(row["reference_text"]),
                device,
                max_steps=args.max_steps,
                stop_threshold=args.stop_threshold,
                attention_window=args.attention_window,
                normalize_wav=args.normalize_wav,
                vocoder=vocoder,
            )
            save_wav(wav_path, wav, sample_rate)
            row["generated_seconds"] = wav.numel() / float(sample_rate)
            row.pop("synthesis_error", None)
        except Exception as exc:  # noqa: BLE001 - keep the long eval resumable.
            row["synthesis_error"] = repr(exc)
            if args.stop_on_error:
                raise


def compute_wer_fields(row: dict[str, Any]) -> None:
    ref_words = normalize_for_wer(str(row.get("reference_text", "")))
    hyp_words = normalize_for_wer(str(row.get("transcript", "")))
    edits = levenshtein_distance(ref_words, hyp_words)
    row["wer_reference_words"] = len(ref_words)
    row["wer_edits"] = edits
    row["wer"] = edits / len(ref_words) if ref_words else None
    row["wer_normalizer"] = "lowercase_punctuation_apostrophe_digits_to_words"


def transcribe_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env}=sk-or-v1-... before using --transcribe.")
    for row in tqdm(entries, desc="Transcribing with OpenRouter"):
        wav_path = Path(row["generated_wav"])
        if not wav_path.exists() or row.get("synthesis_error"):
            continue
        if row.get("transcript") and not args.force_transcribe:
            compute_wer_fields(row)
            continue
        try:
            transcript, response = transcribe_openrouter(
                wav_path,
                api_key=api_key,
                model=args.stt_model,
                endpoint=args.stt_endpoint,
                language=args.language,
                timeout=args.request_timeout,
                retries=args.retries,
                http_referer=args.http_referer,
                x_title=args.x_title,
            )
            row["transcript"] = transcript
            row["openrouter_response"] = response
            row.pop("transcription_error", None)
            compute_wer_fields(row)
        except Exception as exc:  # noqa: BLE001 - keep partial paid eval results.
            row["transcription_error"] = repr(exc)
            if args.stop_on_error:
                raise


def recompute_existing_wer(manifest_path: Path, summary_path: Path, price_per_minute: float) -> dict[str, Any]:
    entries = read_jsonl_rows(manifest_path)
    if not entries:
        raise SystemExit(f"No rows found in {manifest_path}")
    for row in entries:
        if row.get("transcript"):
            compute_wer_fields(row)
    write_jsonl(manifest_path, entries)
    summary = summarize(entries, price_per_minute)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize(entries: list[dict[str, Any]], price_per_minute: float) -> dict[str, Any]:
    generated = [row for row in entries if Path(row["generated_wav"]).exists() and not row.get("synthesis_error")]
    generated_seconds = sum(float(row.get("generated_seconds") or 0.0) for row in generated)
    transcribed = [row for row in generated if "transcript" in row and row.get("wer_reference_words")]
    ref_words = sum(int(row.get("wer_reference_words") or 0) for row in transcribed)
    edits = sum(int(row.get("wer_edits") or 0) for row in transcribed)
    usage_cost = 0.0
    usage_seconds = 0.0
    for row in transcribed:
        usage = row.get("openrouter_response", {}).get("usage", {})
        if isinstance(usage, dict):
            usage_cost += float(usage.get("cost") or 0.0)
            usage_seconds += float(usage.get("seconds") or 0.0)
    return {
        "evaluation_target": "BananaMind V2/V3 generated TTS audio",
        "reference_text_source": "LibriTTS test-clean .normalized.txt",
        "transcription_model": DEFAULT_STT_MODEL,
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
        "estimated_openrouter_cost_usd": generated_seconds / 60.0 * price_per_minute,
        "estimated_openrouter_stt_cost_usd": generated_seconds / 60.0 * price_per_minute,
        "transcribed_examples": len(transcribed),
        "transcription_failures": sum(1 for row in entries if row.get("transcription_error")),
        "wer_reference_words": ref_words,
        "wer_edits": edits,
        "wer": edits / ref_words if ref_words else None,
        "openrouter_reported_cost_usd": usage_cost,
        "openrouter_reported_seconds": usage_seconds,
    }


def print_summary(summary: dict[str, Any], transcribe_requested: bool, yes: bool) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not transcribe_requested:
        print("No OpenRouter calls were made. Add --transcribe --yes to transcribe BananaMind generated WAVs and calculate WER.")
    elif not yes:
        print("OpenRouter calls were not made because --yes was not provided.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Take LibriTTS test-clean text, synthesize it with the BananaMind V2/V3 TTS model, "
            "then optionally transcribe the generated model audio with OpenRouter STT and calculate WER."
        )
    )
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="Path to LibriTTS test-clean tar.gz.")
    parser.add_argument("--limit", type=int, default=1850, help="Number of examples to select from the archive.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated audio, manifests, and summaries.")
    parser.add_argument("--config", default="configs/bananatts_tacotron.yaml")
    parser.add_argument("--checkpoint", default="checkpoints_tacotron/tacotron_latest.pt")
    parser.add_argument("--vocoder", choices=["auto", "griffin-lim", "hifigan"], default="hifigan")
    parser.add_argument("--vocoder-checkpoint", default="checkpoints_vocoder/vocoder_step_42000.pt")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu. Defaults to project auto-detect.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--attention-window", type=int, default=None)
    parser.add_argument("--normalize-wav", action="store_true")
    parser.add_argument("--force-synthesis", action="store_true", help="Regenerate WAVs even if they already exist.")
    parser.add_argument("--metadata-only", action="store_true", help="Only select examples and report source-audio hours from the archive.")
    parser.add_argument("--transcribe", action="store_true", help="Use OpenRouter STT after synthesis.")
    parser.add_argument("--yes", action="store_true", help="Required with --transcribe to confirm paid OpenRouter API use.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--stt-endpoint", default=DEFAULT_STT_ENDPOINT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--price-per-minute", type=float, default=0.003)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--x-title", default="BananaMind-TTS-WER")
    parser.add_argument("--force-transcribe", action="store_true")
    parser.add_argument("--recompute-wer-only", action="store_true", help="Recompute WER from existing saved transcripts without synthesis or API calls.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    archive_path = resolve_path(args.archive)
    out_dir = resolve_path(args.out_dir)
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    samples_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "summary.json"

    if args.recompute_wer_only:
        summary = recompute_existing_wer(manifest_path, summary_path, args.price_per_minute)
        print_summary(summary, transcribe_requested=False, yes=False)
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

    summary = summarize(entries, args.price_per_minute)
    if args.transcribe and not args.yes:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print_summary(summary, transcribe_requested=True, yes=False)
        return

    if args.transcribe:
        transcribe_entries(entries, args)
        write_jsonl(manifest_path, entries)
        summary = summarize(entries, args.price_per_minute)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary, transcribe_requested=args.transcribe, yes=args.yes)


if __name__ == "__main__":
    main()
