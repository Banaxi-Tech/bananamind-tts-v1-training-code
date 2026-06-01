from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bananatts.models.acoustic import FastSpeech2AcousticModel
from bananatts.models.tacotron import TacotronLite
from bananatts.models.vocoder import HiFiGANGenerator
from bananatts.text import TextTokenizer
from bananatts.utils import count_parameters, format_param_count, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Count BananaTTS model parameters.")
    parser.add_argument("--config", default="configs/bananatts_20m.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    tokenizer = TextTokenizer.from_config(config["text"])
    if config["model"].get("type") == "tacotron_lite":
        acoustic = TacotronLite(tokenizer.vocab_size, int(config["audio"]["n_mels"]), config["model"], tokenizer.pad_id)
    else:
        acoustic = FastSpeech2AcousticModel(
            tokenizer.vocab_size,
            int(config["audio"]["n_mels"]),
            config["model"],
            tokenizer.pad_id,
            special_token_ids=(tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id),
        )
    acoustic_params = count_parameters(acoustic)
    print(f"Acoustic model: {format_param_count(acoustic_params)} ({acoustic_params:,})")
    if config.get("vocoder", {}).get("type", "").replace("-", "_").lower() in {"hifigan", "hifi_gan"}:
        vocoder = HiFiGANGenerator(int(config["audio"]["n_mels"]), config["vocoder"])
        print(f"V3 HiFi-GAN generator: {format_param_count(count_parameters(vocoder))} ({count_parameters(vocoder):,})")
    else:
        print("Vocoder model: Griffin-Lim fallback (0 trainable parameters)")


if __name__ == "__main__":
    main()
