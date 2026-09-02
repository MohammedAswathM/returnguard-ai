#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.data.validation import validate_uci_archive

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, default=Path("data/raw/online_retail_ii.zip"))
parser.add_argument("--output", type=Path, default=Path("artifacts/source/uci_validation.json"))
args = parser.parse_args()
report = validate_uci_archive(args.archive, args.output)
print(json.dumps({"status": report["status"], "sha256": report["archive"]["sha256"],
                  "rows": report["totals"]["rows"], "report": str(args.output)}))
