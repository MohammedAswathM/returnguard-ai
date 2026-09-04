#!/usr/bin/env python3
import json
from pathlib import Path

from returnguard.data.fingerprints import file_sha256
from returnguard.v2.audit import run_shortcut_audit
from returnguard.v2.config import load_v2_feature_config
from returnguard.v2.features import assert_ordered_schema, build_features

data_dir = Path("artifacts/v2/data")
output_dir = Path("artifacts/v2/development")
output_dir.mkdir(parents=True, exist_ok=True)
config = load_v2_feature_config(Path("configs/v2/features.yaml"))
features = build_features(data_dir, config)
assert_ordered_schema(features, config)
feature_path = output_dir / "features.parquet"
features.to_parquet(feature_path, index=False)
audit = run_shortcut_audit(features, config, output_dir / "shortcut_audit.json")
print(json.dumps({
    "support": len(features),
    "feature_sha256": file_sha256(feature_path),
    "shortcut_audit_passed": audit["passed"],
}))
