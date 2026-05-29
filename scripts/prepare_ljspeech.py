from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bananatts.data import prepare_ljspeech
from bananatts.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LJSpeech features for BananaTTS.")
    parser.add_argument("--config", default="configs/bananatts_20m.yaml")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name, e.g. MikhailT/lj-speech")
    parser.add_argument("--local-path", default=None, help="Path to extracted official LJSpeech-1.1 directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--percent", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    local_path = None
    if args.local_path:
        local_path = Path(args.local_path)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
        prepare_ljspeech(
            config,
            local_path=local_path,
            limit=args.limit,
            percent=args.percent,
            force=args.force,
        )
        return

    try:
        prepare_ljspeech(config, dataset_name=args.dataset, limit=args.limit, percent=args.percent, force=args.force)
    except Exception:
        fallback = config["dataset"].get("fallback_name")
        requested = args.dataset or config["dataset"]["name"]
        if fallback and fallback != requested:
            print(f"Dataset {requested} failed; retrying {fallback}")
            prepare_ljspeech(config, dataset_name=fallback, limit=args.limit, percent=args.percent, force=args.force)
        else:
            raise


if __name__ == "__main__":
    main()
