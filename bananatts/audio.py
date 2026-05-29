from __future__ import annotations

import math
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def to_mono(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    wav = torch.as_tensor(audio, dtype=torch.float32)
    if wav.ndim == 2:
        if wav.shape[0] <= 8:
            wav = wav.mean(dim=0)
        else:
            wav = wav.mean(dim=1)
    if wav.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {tuple(wav.shape)}")
    return wav.contiguous()


def resample_linear(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    if wav.numel() == 0:
        return wav
    new_len = max(1, int(round(wav.numel() * target_sr / orig_sr)))
    wav_3d = wav.view(1, 1, -1)
    return F.interpolate(wav_3d, size=new_len, mode="linear", align_corners=False).view(-1)


def preemphasis_safe(wav: torch.Tensor) -> torch.Tensor:
    wav = torch.nan_to_num(wav.float())
    peak = wav.abs().max()
    if peak > 1.0:
        wav = wav / peak
    return wav.clamp(-1.0, 1.0)


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    f_max = min(float(f_max), sample_rate / 2)
    m_min = hz_to_mel(torch.tensor(float(f_min)))
    m_max = hz_to_mel(torch.tensor(float(f_max)))
    m_points = torch.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    bins = torch.floor((n_fft + 1) * f_points / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i].item(), bins[i + 1].item(), bins[i + 2].item()
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, fb.shape[1])):
            fb[i, j] = (j - left) / max(1, center - left)
        for j in range(center, min(right, fb.shape[1])):
            fb[i, j] = (right - j) / max(1, right - center)
    return fb


class MelSpectrogram:
    def __init__(self, config: dict[str, Any]):
        self.sample_rate = int(config["sample_rate"])
        self.n_fft = int(config["n_fft"])
        self.hop_length = int(config["hop_length"])
        self.win_length = int(config["win_length"])
        self.n_mels = int(config["n_mels"])
        self.f_min = float(config.get("f_min", 0.0))
        self.f_max = float(config.get("f_max", self.sample_rate / 2))
        self.power = float(config.get("power", 1.0))
        self.window = torch.hann_window(self.win_length)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        wav = preemphasis_safe(wav)
        if wav.numel() < self.win_length:
            wav = F.pad(wav, (0, self.win_length - wav.numel()))
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(wav.device),
            return_complex=True,
            center=True,
        )
        mag = spec.abs().pow(self.power)
        fb = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels, self.f_min, self.f_max).to(mag.device)
        mel = fb @ mag
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(0, 1).contiguous()


def mel_to_linear_magnitude(mel: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    mel = mel.float()
    audio_cfg = config
    fb = mel_filterbank(
        int(audio_cfg["sample_rate"]),
        int(audio_cfg["n_fft"]),
        int(audio_cfg["n_mels"]),
        float(audio_cfg.get("f_min", 0.0)),
        float(audio_cfg.get("f_max", int(audio_cfg["sample_rate"]) / 2)),
    ).to(mel.device)
    mel_mag = torch.exp(mel).transpose(0, 1)
    return torch.linalg.pinv(fb) @ mel_mag


def griffin_lim(mel: torch.Tensor, config: dict[str, Any], n_iters: int | None = None) -> torch.Tensor:
    n_fft = int(config["n_fft"])
    hop_length = int(config["hop_length"])
    win_length = int(config["win_length"])
    n_iters = int(n_iters or config.get("griffin_lim_iters", 32))
    window = torch.hann_window(win_length, device=mel.device)
    mag = mel_to_linear_magnitude(mel, config).clamp_min(1e-6)
    angle = 2 * math.pi * torch.rand_like(mag)
    spec = torch.polar(mag, angle)
    wav_len = max(1, (mel.shape[0] - 1) * hop_length)
    wav = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=wav_len)
    for _ in range(n_iters):
        rebuilt = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        spec = mag * torch.exp(1j * rebuilt.angle())
        wav = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=wav_len)
    return preemphasis_safe(wav.detach().cpu())


def save_wav(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_np = preemphasis_safe(wav).detach().cpu().numpy()
    wav_i16 = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wav_i16.tobytes())


def validate_audio(wav: torch.Tensor, sample_rate: int, min_seconds: float, max_seconds: float) -> None:
    if wav.numel() == 0:
        raise ValueError("empty audio")
    seconds = wav.numel() / sample_rate
    if seconds < min_seconds:
        raise ValueError(f"audio too short: {seconds:.3f}s")
    if seconds > max_seconds:
        raise ValueError(f"audio too long: {seconds:.3f}s")
    if not torch.isfinite(wav).all():
        raise ValueError("audio contains NaN or inf")
