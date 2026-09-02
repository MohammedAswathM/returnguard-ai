from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from returnguard.domain.enums import VerificationResult


def estimate_likelihoods(
    requests: pd.DataFrame, verifications: pd.DataFrame, alpha: float,
    inconclusive_bounds: tuple[float, float] = (0.8, 1.25),
) -> dict[str, Any]:
    allowed = {"train", "calibration"}
    request_rows = requests.loc[requests["partition"].isin(allowed), [
        "refund_request_id", "is_refund_abuse_simulated",
    ]]
    verification_rows = verifications.loc[verifications["partition"].isin(allowed), [
        "refund_request_id", "result",
    ]]
    merged = request_rows.merge(verification_rows, on="refund_request_id", validate="one_to_one")
    results = [result.value for result in VerificationResult]
    counts: dict[str, dict[str, int]] = {"abusive": {}, "legitimate": {}}
    empirical_likelihood_ratios: dict[str, float] = {}
    conditional: dict[str, dict[str, float]] = {"abusive": {}, "legitimate": {}}
    for truth_name, truth_value in (("abusive", True), ("legitimate", False)):
        subset = merged.loc[merged["is_refund_abuse_simulated"].astype(bool) == truth_value]
        denominator = len(subset) + alpha * len(results)
        for result in results:
            count = int((subset["result"] == result).sum())
            counts[truth_name][result] = count
            conditional[truth_name][result] = float((count + alpha) / denominator)
    for result in results:
        empirical_likelihood_ratios[result] = (
            conditional["abusive"][result] / conditional["legitimate"][result]
        )
    if empirical_likelihood_ratios["consistent"] >= 1:
        raise ValueError("consistent likelihood ratio must reduce risk")
    if empirical_likelihood_ratios["inconsistent"] <= 1:
        raise ValueError("inconsistent likelihood ratio must increase risk")
    applied = dict(empirical_likelihood_ratios)
    applied["inconclusive"] = float(
        np.clip(applied["inconclusive"], inconclusive_bounds[0], inconclusive_bounds[1])
    )
    applied["expired"] = 1.0
    return {
        "version": "order-integrity-likelihoods-v1", "partitions": sorted(allowed),
        "laplace_alpha": alpha, "support_counts": counts,
        "conditional_probabilities": conditional,
        "empirical_likelihood_ratios": empirical_likelihood_ratios,
        "likelihood_ratios": applied,
        "safety_overrides": {
            "inconclusive_bounds": list(inconclusive_bounds),
            "expired_is_neutral": True,
        },
    }


def bayesian_update(prior_probability: float, likelihood_ratio: float) -> float:
    prior = float(np.clip(prior_probability, 1e-6, 1 - 1e-6))
    prior_odds = prior / (1 - prior)
    posterior_odds = prior_odds * likelihood_ratio
    return float(np.clip(posterior_odds / (1 + posterior_odds), 1e-6, 1 - 1e-6))
