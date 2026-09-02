import json
from pathlib import Path
from typing import Any

import pandas as pd

from returnguard.data.fingerprints import file_sha256


def seal_final_truth(
    requests: pd.DataFrame, verifications: pd.DataFrame, lock_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    final_requests = requests.loc[requests["partition"] == "final_test", [
        "refund_request_id", "is_refund_abuse_simulated", "simulation_scenario",
        "outcome_available_at",
    ]]
    final_verifications = verifications.loc[verifications["partition"] == "final_test", [
        "refund_request_id", "completed_at", "result", "result_codes",
    ]]
    truth = final_requests.merge(
        final_verifications, on="refund_request_id", how="left", validate="one_to_one"
    )
    truth_path = lock_dir / "final_truth.csv"
    truth.to_csv(truth_path, index=False, lineterminator="\n")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "state": "SEALED",
        "support": len(truth),
        "truth_sha256": file_sha256(truth_path),
        "opened_at": None,
        "invalidation_reason": None,
    }
    (lock_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    public_requests = requests.copy()
    public_requests["is_refund_abuse_simulated"] = public_requests[
        "is_refund_abuse_simulated"
    ].astype("boolean")
    final_request_mask = public_requests["partition"] == "final_test"
    for column in (
        "is_refund_abuse_simulated", "simulation_scenario", "outcome_available_at",
    ):
        public_requests.loc[final_request_mask, column] = pd.NA
    public_verifications = verifications.copy()
    final_verification_mask = public_verifications["partition"] == "final_test"
    for column in ("completed_at", "result", "result_codes"):
        public_verifications.loc[final_verification_mask, column] = None
    return public_requests, public_verifications, manifest
