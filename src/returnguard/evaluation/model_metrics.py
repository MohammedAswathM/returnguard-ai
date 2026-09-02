from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from returnguard.evaluation.metrics import classification_metrics


def reliability_bins(
    labels: npt.NDArray[np.bool_], probabilities: npt.NDArray[np.float64], bins: int,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        result.append({
            "bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "support": int(mask.sum()),
            "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
            "observed_prevalence": float(labels[mask].mean()) if mask.any() else None,
        })
    return result


def model_metrics(
    frame: pd.DataFrame, probabilities: npt.NDArray[np.float64], threshold: float, bins: int,
) -> dict[str, Any]:
    labels = frame["is_refund_abuse_simulated"].astype(bool).to_numpy()
    report = classification_metrics(labels, probabilities, threshold)
    report["roc_auc"] = float(roc_auc_score(labels, probabilities))
    report["brier_score"] = float(brier_score_loss(labels, probabilities))
    abusive_amount = frame.loc[labels, "requested_amount_paise"].sum()
    caught_amount = frame.loc[labels & (probabilities >= threshold), "requested_amount_paise"].sum()
    report["abuse_amount_recall"] = (
        float(caught_amount / abusive_amount) if abusive_amount else 0.0
    )
    report["reliability_bins"] = reliability_bins(labels, probabilities, bins)
    return report
