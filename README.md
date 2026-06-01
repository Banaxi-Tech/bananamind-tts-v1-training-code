# BananaTTS-20M

BananaTTS-20M is a from-scratch, single-speaker English TTS training pipeline for LJSpeech. It is intended to be a clear first working system, not a state-of-the-art voice model.

This project trains one fixed English voice from text to mel spectrograms with a compact FastSpeech2-style acoustic model. V3 adds a self-trained HiFi-GAN vocoder for better waveform quality, while Griffin-Lim remains available as a debugging fallback.

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

V3 vocoder training needs waveform targets in the prepared tensors. If your cache was created before V3, rebuild it once:

```bash
python scripts/prepare_ljspeech.py --local-path data/raw/LJSpeech-1.1 --force
```

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

## Train V3 HiFi-GAN Vocoder

The V3 vocoder is trained from scratch on the same prepared LJSpeech cache. It learns to convert the project log-mel features into waveform audio and replaces Griffin-Lim during synthesis.

Small debug run:

```bash
python -m bananatts.train_vocoder \
  --config configs/bananatts_v3_hifigan.yaml \
  --prepare \
  --local-path data/raw/LJSpeech-1.1 \
  --limit 1000
```

Full run after the cache exists:

```bash
python -m bananatts.train_vocoder --config configs/bananatts_v3_hifigan.yaml
```

The latest generator checkpoint is saved at `checkpoints_vocoder/vocoder_latest.pt`.

## Synthesize

With a trained checkpoint:

```bash
python -m bananatts.synthesize \
  --config configs/bananatts_20m.yaml \
  --checkpoint checkpoints/acoustic_latest.pt \
  --text "Banana TTS is a small text to speech model." \
  --out samples/test.wav
```

With the V3 HiFi-GAN vocoder:

```bash
python -m bananatts.synthesize \
  --config configs/bananatts_v3_hifigan.yaml \
  --checkpoint checkpoints/acoustic_latest.pt \
  --vocoder hifigan \
  --vocoder-checkpoint checkpoints_vocoder/vocoder_latest.pt \
  --text "Banana TTS now uses a self trained HiFi GAN vocoder." \
  --out samples/test_v3.wav \
  --normalize-wav
```

Tacotron-lite can use the same V3 vocoder checkpoint:

```bash
python -m bananatts.synthesize_tacotron \
  --config configs/bananatts_tacotron.yaml \
  --checkpoint checkpoints_tacotron/tacotron_latest.pt \
  --vocoder hifigan \
  --vocoder-checkpoint checkpoints_vocoder/vocoder_latest.pt \
  --text "Hello from Banana TTS V3." \
  --out samples/tacotron_v3.wav \
  --normalize-wav
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

The current acoustic model is intentionally compact. The V3 inference stack is roughly 26M parameters with the default acoustic model and HiFi-GAN generator.

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
- compact HiFi-GAN generator with multi-period and multi-scale discriminators
- self-trained from prepared LJSpeech waveform/mel pairs

## Expected Limitations

- Uniform durations are a crude target and limit quality.
- Griffin-Lim audio sounds rough and buzzy compared with the V3 neural vocoder.
- Character input is less robust than phonemes for English pronunciation.
- Training from scratch on LJSpeech needs real training time before intelligible speech appears.
- No pretrained aligner, acoustic model, or vocoder is used.

## Hardware Notes

For a 16 GB NVIDIA GPU:

- Start with `batch_size: 12` at 22.05 kHz.
- Lower `batch_size` to 4-8 if clips or gradients exceed memory.
- Mixed precision is enabled automatically on CUDA.
- Use `--limit 1000` first to verify throughput and checkpoints.
- Full LJSpeech acoustic training should fit. V3 HiFi-GAN training is more GPU-sensitive than acoustic training; lower `vocoder_training.batch_size` first if memory is tight.

## Useful Commands

```bash
python scripts/prepare_ljspeech.py --dataset MikhailT/lj-speech --limit 1000
python scripts/smoke_test.py
python -m bananatts.train_acoustic --config configs/bananatts_20m.yaml --limit 1000
python -m bananatts.train_vocoder --config configs/bananatts_v3_hifigan.yaml --limit 1000 --prepare
python -m bananatts.synthesize --config configs/bananatts_20m.yaml --text "Hello from Banana TTS." --out samples/hello.wav
```
