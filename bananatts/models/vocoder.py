from __future__ import annotations

from typing import Any

import torch

from ..audio import griffin_lim


class GriffinLimVocoder:
    """Debug vocoder fallback.

    This is not production TTS quality. It exists so the acoustic model path can
    produce a WAV before a small HiFiGAN-style vocoder is implemented.
    """

    def __init__(self, audio_config: dict[str, Any]):
        self.audio_config = audio_config

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        return griffin_lim(mel, self.audio_config)


class TinyHiFiGANPlaceholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "TODO: implement a small HiFiGAN-style generator and discriminators. "
            "Use GriffinLimVocoder for v0.1 debugging synthesis."
        )
