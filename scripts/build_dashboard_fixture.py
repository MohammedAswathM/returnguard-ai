#!/usr/bin/env python3
import json
from pathlib import Path

from returnguard.dashboard.app import build_demo_case_payload

path = Path("artifacts/dashboard/demo_case_payload.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(build_demo_case_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"path": path.as_posix(), "status": "SANITIZED_DEMO_FIXTURE_CREATED"}))
