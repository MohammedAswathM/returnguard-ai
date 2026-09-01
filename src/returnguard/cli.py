import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from returnguard.data.download import download_uci
from returnguard.pipeline import generate, train_baselines


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="returnguard")
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("generate-data")
    data.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    data.add_argument("--output-dir", type=Path)
    train = commands.add_parser("train-baselines")
    train.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    train.add_argument("--config", type=Path, default=Path("configs/model_baseline.yaml"))
    train.add_argument("--features", type=Path, default=Path("configs/features.yaml"))
    download = commands.add_parser("download-uci")
    download.add_argument("--output", type=Path, default=Path("data/raw/online_retail_ii.zip"))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "generate-data":
        result = generate(args.config, args.output_dir)
        print(json.dumps({"fingerprints": result["fingerprints"], "splits": result["split_summary"]}))
    elif args.command == "train-baselines":
        result = train_baselines(args.data_dir, args.config, args.features)
        print(json.dumps(result["sanity_checks"]))
    else:
        print(download_uci(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

