#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from returnguard.data.fingerprints import file_sha256, object_sha256

results_dir = Path("artifacts/final_results")
cases_path = results_dir / "final_case_results.parquet"
results = json.loads((results_dir / "results.lock.json").read_text(encoding="utf-8"))
if file_sha256(cases_path) != results["case_results_sha256"]:
    raise ValueError("final case artifact does not match locked results")
cases = pd.read_parquet(cases_path)
requests = pd.read_csv(
    results_dir / "opened_data" / "refund_requests.csv",
    usecols=["refund_request_id", "simulation_scenario"],
)
cold = pd.read_csv("artifacts/data_uci/cold_start_mapping.csv")
frame = cases.merge(requests, on="refund_request_id", validate="one_to_one")


def summarize(selected: pd.DataFrame) -> dict[str, Any]:
    labels = selected["label_simulated"].astype(bool)
    return {
        "support": len(selected), "positive_support": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(selected) else None,
        "average_probability": float(selected["calibrated_probability"].mean()) if len(selected) else None,
        "average_precision": (
            float(average_precision_score(labels, selected["calibrated_probability"]))
            if labels.nunique() == 2 else None
        ),
        "underpowered": bool(len(selected) < 50 or labels.sum() < 10),
    }


hard_scenarios = (
    "shared_household", "loyal_high_volume", "product_defect_burst", "carrier_incident",
    "legitimate_high_value_damage", "first_order_return", "missing_evidence_inconclusive",
)
payload: dict[str, Any] = {
    "schema_version": "1.0", "source_case_results_sha256": results["case_results_sha256"],
    "cold_start_challenge": summarize(frame.loc[
        frame["refund_request_id"].isin(cold["refund_request_id"])
    ]),
    "hard_legitimate_scenarios": {
        scenario: summarize(frame.loc[frame["simulation_scenario"] == scenario])
        for scenario in hard_scenarios
    },
    "timing_disclosure": (
        "Slice labels were summarized only after the frozen one-shot final evaluation; "
        "they were not used for fitting or policy selection."
    ),
}
payload["slice_results_sha256"] = object_sha256(payload)
Path("slice_results.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
