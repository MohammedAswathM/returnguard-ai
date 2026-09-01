from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, confusion_matrix


def classification_metrics(
    labels: npt.NDArray[np.bool_], scores: npt.NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    return {
        "support": int(len(labels)), "prevalence": float(np.mean(labels)),
        "threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "average_precision": float(average_precision_score(labels, scores)),
    }
