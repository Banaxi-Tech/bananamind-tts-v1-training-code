from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a future BananaTTS HiFiGAN-style vocoder.")
    parser.add_argument("--config", default="configs/bananatts_20m.yaml")
    parser.parse_args()
    raise NotImplementedError(
        "Vocoder training is intentionally a TODO in v0.1. "
        "The current pipeline uses Griffin-Lim fallback for debugging synthesis."
    )


if __name__ == "__main__":
    main()
