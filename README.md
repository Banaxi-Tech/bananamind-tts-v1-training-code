# BananaTTS-20M

BananaTTS-20M is a from-scratch, single-speaker English TTS training pipeline for LJSpeech. It is intended to be a clear first working system, not a state-of-the-art voice model.

This project trains one fixed English voice from text to mel spectrograms with a compact FastSpeech2-style acoustic model. The first version uses a Griffin-Lim vocoder fallback so synthesis can be debugged before a small HiFiGAN-style vocoder is implemented.

## What This Is Not

- Not a voice cloning system.
- No speaker embeddings.
- No reference audio conditioning.
- No pretrained TTS checkpoints.
- No multi-speaker training path.

## Install

```bash
cd banana-tts-20m
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code avoids making `torchaudio` mandatory for the core mel path, but `soundfile` or `torchaudio` is useful for Hugging Face audio decoding depending on your environment.

## Prepare LJSpeech

Default dataset:

```bash
python scripts/prepare_ljspeech.py --dataset MikhailT/lj-speech --limit 1000
```

Fallback dataset:

```bash
python scripts/prepare_ljspeech.py --dataset keithito/lj_speech --limit 1000
```

Prepared tensors, manifests, split files, and feature config are stored under `data/cache/ljspeech_22050`.

## Smoke Test

```bash
python scripts/smoke_test.py
```

The smoke test prepares or loads 10 LJSpeech samples when possible, builds the model, runs one forward pass, runs one optimization step, prints the parameter count, and writes `samples/smoke_test.wav`. If local audio decoding dependencies are missing, it falls back to synthetic tensors for shape validation and reports that clearly.

## Train Acoustic Model

Small debug run:

```bash
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml --limit 1000
```

Train on 50% of LJSpeech:

```bash
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml --percent 50
```

Train on full LJSpeech:

```bash
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml
```

## Train Tacotron-Lite Acoustic Model

The FastSpeech-style path needs real duration alignments. If you hear the model start a word and then collapse into repeated noise, train the Tacotron-lite path instead. It learns attention alignment directly from text and mel frames.

Prepare the normalized cache first:

```bash
python scripts/prepare_ljspeech.py --local-path data/raw/LJSpeech-1.1 --force
```

Train:

```bash
python -m bananatts.train_tacotron --config configs/bananatts_tacotron.yaml
```

Synthesize:

```bash
python -m bananatts.synthesize_tacotron \
  --config configs/bananatts_tacotron.yaml \
  --checkpoint checkpoints_tacotron/tacotron_latest.pt \
  --text "Hello from Banana TTS. This is a simple speech test." \
  --out samples/tacotron_test.wav \
  --normalize-wav \
  --debug
```

Resume:

```bash
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml --resume checkpoints/acoustic_latest.pt
```

## Synthesize

With a trained checkpoint:

```bash
python -m bananatts.synthesize \
  --config configs/bananatts_20m.yaml \
  --checkpoint checkpoints/acoustic_latest.pt \
  --text "Banana TTS is a small text to speech model." \
  --out samples/test.wav
```

Pipeline debug without a checkpoint:

```bash
python -m bananatts.synthesize --config configs/bananatts_20m.yaml --text "Hello from Banana TTS." --out samples/hello.wav
```

Without a checkpoint, the model uses random weights, so the output is only useful for verifying that synthesis writes a WAV.

## Parameter Count

```bash
python scripts/count_params.py
```

The current acoustic model is intentionally compact. The eventual target is around 20M inference parameters total after adding a small neural vocoder.

## Architecture

Current acoustic model:

- character tokenizer with optional future phoneme hook
- token embedding
- sinusoidal positional encoding
- 4-layer Transformer encoder
- convolutional duration predictor
- uniform-duration training targets as a first-pass alignment substitute
- length regulator
- 4-layer Transformer decoder
- mel projection

Current vocoder:

- Griffin-Lim fallback for debugging
- `train_vocoder.py` is a clear TODO for a small HiFiGAN-style vocoder

## Expected Limitations

- Uniform durations are a crude target and limit quality.
- Griffin-Lim audio sounds rough and buzzy compared with a neural vocoder.
- Character input is less robust than phonemes for English pronunciation.
- Training from scratch on LJSpeech needs real training time before intelligible speech appears.
- No pretrained aligner, acoustic model, or vocoder is used.

## Hardware Notes

For a 16 GB NVIDIA GPU:

- Start with `batch_size: 12` at 22.05 kHz.
- Lower `batch_size` to 4-8 if clips or gradients exceed memory.
- Mixed precision is enabled automatically on CUDA.
- Use `--limit 1000` first to verify throughput and checkpoints.
- Full LJSpeech acoustic training should fit, but Griffin-Lim synthesis is CPU-heavy and neural-vocoder training is not implemented yet.

## Useful Commands

```bash
python scripts/prepare_ljspeech.py --dataset MikhailT/lj-speech --limit 1000
python scripts/smoke_test.py
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml --limit 1000
python -m bananatts.synthesize --config configs/bananatts_20m.yaml --text "Hello from Banana TTS." --out samples/hello.wav
```
