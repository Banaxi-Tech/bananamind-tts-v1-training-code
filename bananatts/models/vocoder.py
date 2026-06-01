from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..audio import griffin_lim


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GriffinLimVocoder:
    """Debug vocoder fallback.

    This is not production TTS quality. It stays available so the acoustic
    model path can produce a WAV without a trained V3 HiFi-GAN checkpoint.
    """

    def __init__(self, audio_config: dict[str, Any]):
        self.audio_config = audio_config

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        return griffin_lim(mel, self.audio_config)


class HiFiGANResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int], slope: float = 0.1):
        super().__init__()
        self.slope = slope
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=(kernel_size * dilation - dilation) // 2,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = x
            y = F.leaky_relu(x, self.slope)
            y = conv1(y)
            y = F.leaky_relu(y, self.slope)
            y = conv2(y)
            x = y + residual
        return x


class HiFiGANGenerator(nn.Module):
    """Compact HiFi-GAN generator for BananaTTS log-mel features.

    The acoustic pipeline uses center-padded STFT mels where a mel with T frames
    maps naturally to roughly ``(T - 1) * hop_length`` waveform samples. The
    generator upsamples by ``hop_length`` and callers crop to that target length.
    """

    def __init__(self, n_mels: int, config: dict[str, Any]):
        super().__init__()
        channels = int(config.get("initial_channels", 256))
        upsample_rates = [int(x) for x in config.get("upsample_rates", [8, 8, 2, 2])]
        upsample_kernels = [int(x) for x in config.get("upsample_kernel_sizes", [16, 16, 4, 4])]
        resblock_kernels = [int(x) for x in config.get("resblock_kernel_sizes", [3, 7, 11])]
        resblock_dilations = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        self.slope = float(config.get("leaky_relu_slope", 0.1))
        self.conv_pre = nn.Conv1d(n_mels, channels, kernel_size=7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        current_channels = channels
        for rate, kernel in zip(upsample_rates, upsample_kernels, strict=True):
            next_channels = current_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    current_channels,
                    next_channels,
                    kernel_size=kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            for kernel_size, dilations in zip(resblock_kernels, resblock_dilations, strict=True):
                self.resblocks.append(
                    HiFiGANResBlock(next_channels, int(kernel_size), [int(d) for d in dilations], self.slope)
                )
            current_channels = next_channels

        self.num_resblocks = len(resblock_kernels)
        self.conv_post = nn.Conv1d(current_channels, 1, kernel_size=7, padding=3)
        self.apply(_init_weights)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"Expected mel tensor [B, n_mels, T], got {tuple(mel.shape)}")
        x = self.conv_pre(mel)
        for idx, upsample in enumerate(self.ups):
            x = F.leaky_relu(x, self.slope)
            x = upsample(x)
            block_offset = idx * self.num_resblocks
            xs = [
                self.resblocks[block_offset + block_idx](x)
                for block_idx in range(self.num_resblocks)
            ]
            x = torch.stack(xs, dim=0).mean(dim=0)
        x = F.leaky_relu(x, self.slope)
        return torch.tanh(self.conv_post(x)).squeeze(1)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, slope: float = 0.1):
        super().__init__()
        self.period = period
        self.slope = slope
        channels = [(1, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024)]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
                for in_ch, out_ch in channels
            ]
        )
        self.conv_post = nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))
        self.apply(_init_weights)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        batch, channels, time = wav.shape
        if time % self.period:
            wav = F.pad(wav, (0, self.period - (time % self.period)), mode="reflect")
            time = wav.shape[-1]
        x = wav.view(batch, channels, time // self.period, self.period)
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.slope)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] | None = None):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in (periods or [2, 3, 5, 7, 11])])

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        return [disc(wav) for disc in self.discriminators]


class ScaleDiscriminator(nn.Module):
    def __init__(self, slope: float = 0.1):
        super().__init__()
        self.slope = slope
        specs = [
            (1, 32, 15, 1, 7),
            (32, 128, 41, 2, 20),
            (128, 256, 41, 2, 20),
            (256, 512, 41, 4, 20),
            (512, 512, 41, 4, 20),
            (512, 512, 5, 1, 2),
        ]
        self.convs = nn.ModuleList([nn.Conv1d(*spec) for spec in specs])
        self.conv_post = nn.Conv1d(512, 1, kernel_size=3, padding=1)
        self.apply(_init_weights)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        features = []
        x = wav
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.slope)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator(), ScaleDiscriminator(), ScaleDiscriminator()])
        self.pools = nn.ModuleList([nn.Identity(), nn.AvgPool1d(4, stride=2, padding=1), nn.AvgPool1d(4, stride=2, padding=1)])

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        if wav.ndim == 2:
            x = wav.unsqueeze(1)
        else:
            x = wav
        outputs = []
        for pool, disc in zip(self.pools, self.discriminators, strict=True):
            pooled = pool(x)
            outputs.append(disc(pooled))
        return outputs


def discriminator_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    loss = real_outputs[0][0].new_tensor(0.0)
    for (real_score, _), (fake_score, _) in zip(real_outputs, fake_outputs, strict=True):
        loss = loss + torch.mean((real_score - 1.0) ** 2) + torch.mean(fake_score**2)
    return loss


def generator_loss(fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]]) -> torch.Tensor:
    loss = fake_outputs[0][0].new_tensor(0.0)
    for fake_score, _ in fake_outputs:
        loss = loss + torch.mean((fake_score - 1.0) ** 2)
    return loss


def feature_matching_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    loss = real_outputs[0][0].new_tensor(0.0)
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs, strict=True):
        for real_feature, fake_feature in zip(real_features, fake_features, strict=True):
            loss = loss + F.l1_loss(fake_feature, real_feature.detach())
    return loss


def crop_audio_to_mel_length(wav: torch.Tensor, mel_frames: int, hop_length: int) -> torch.Tensor:
    target_samples = max(1, (int(mel_frames) - 1) * int(hop_length))
    if wav.shape[-1] < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.shape[-1]))
    return wav[..., :target_samples]


class HiFiGANVocoder:
    def __init__(
        self,
        audio_config: dict[str, Any],
        vocoder_config: dict[str, Any],
        checkpoint: str | Path,
        device: torch.device | str,
    ):
        self.audio_config = audio_config
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        ckpt_config = ckpt.get("config")
        if isinstance(ckpt_config, dict) and isinstance(ckpt_config.get("vocoder"), dict):
            vocoder_config = ckpt_config["vocoder"]
        self.generator = HiFiGANGenerator(int(audio_config["n_mels"]), vocoder_config).to(self.device)
        state = ckpt.get("generator", ckpt.get("model", ckpt))
        self.generator.load_state_dict(state)
        self.generator.eval()

    @torch.no_grad()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel_batch = mel.float().transpose(0, 1).unsqueeze(0).to(self.device)
        wav = self.generator(mel_batch)
        wav = crop_audio_to_mel_length(wav, mel.shape[0], int(self.audio_config["hop_length"]))
        return wav.squeeze(0).detach().cpu()


def load_vocoder_from_config(
    config: dict[str, Any],
    device: torch.device | str,
    checkpoint: str | Path | None = None,
    vocoder_type: str | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    audio_config = config["audio"]
    vocoder_config = dict(config.get("vocoder", {}))
    selected_type = (vocoder_type or vocoder_config.get("type") or "griffin_lim").replace("-", "_").lower()
    if selected_type in {"griffin_lim", "griffinlim", "gl"}:
        return GriffinLimVocoder(audio_config)
    if selected_type in {"hifigan", "hifi_gan", "hifi-gan"}:
        checkpoint = checkpoint or vocoder_config.get("checkpoint")
        if not checkpoint:
            raise FileNotFoundError("HiFi-GAN vocoder selected but no --vocoder-checkpoint or vocoder.checkpoint was provided.")
        return HiFiGANVocoder(audio_config, vocoder_config, checkpoint, device)
    raise ValueError(f"Unknown vocoder type: {selected_type}")
