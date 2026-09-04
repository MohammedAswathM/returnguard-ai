#!/usr/bin/env python3
import json
from pathlib import Path

from returnguard.data.fingerprints import file_sha256
from returnguard.v2.config import load_v2_feature_config

descriptions = {
    "account_tenure_days": "Customer account age at the claim decision.",
    "requested_amount_paise": "Current technically valid claimed refund amount.",
    "requested_quantity_ratio": "Claimed quantity divided by purchased quantity.",
    "hours_delivery_to_request": "Elapsed hours from delivery to claim submission.",
    "reason_code": "Structured merchant claim reason.",
    "category": "Source-derived product category group.",
    "prior_order_count": "Orders placed strictly before the claim decision.",
    "smoothed_refund_rate_90d": "Empirically smoothed prior 90-day claim rate.",
    "prior_matured_adverse_rate": "Smoothed adverse rate among outcomes matured before decision.",
    "decayed_refund_velocity_30d": "Exponentially decayed prior 30-day claim activity.",
    "customer_peer_refund_deviation": "Customer smoothed claim rate minus the prior portfolio rate.",
    "category_matured_defect_rate": "Smoothed matured legitimate-defect rate for the category.",
    "reason_history_inconsistency": "Deviation of the current reason from prior customer reasons.",
    "evidence_provided": "Whether structured evidence is present for this claim.",
    "returnless_requested": "Whether the current request asks for returnless treatment.",
    "prior_connected_claim_count": "Prior claims sharing the current address or device.",
    "connected_matured_adverse_rate": "Smoothed matured adverse rate for supporting entities.",
    "missing_evidence_indicator": "Explicit missingness indicator for structured evidence.",
    "cold_start_indicator": "Whether no prior claim is available for this customer.",
}
config_path = Path("configs/v2/features.yaml")
config = load_v2_feature_config(config_path)
rows = []
for spec in config.features:
    rows.append({
        "name": spec.name, "dtype": spec.type, "default": spec.default,
        "group": spec.group, "semantic_description": descriptions[spec.name],
        "cutoff_rule": spec.cutoff_rule, "maturity_rule": spec.maturity_rule,
        "explanation_mapping": {
            "reason_code": f"V2_{spec.name.upper()}",
            "template": f"Model input {spec.name} differed from its development reference.",
            "interpretation_boundary": "Model association only; not a causal or guilt claim.",
        },
    })
report = {
    "schema_version": "2.0", "feature_config_sha256": file_sha256(config_path),
    "ordered_features": rows, "forbidden_features": list(config.forbidden_features),
}
path = Path("evidence/v2/feature_dictionary.json")
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"path": path.as_posix(), "sha256": file_sha256(path)}))
