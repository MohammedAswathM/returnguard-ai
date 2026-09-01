import numpy as np

from returnguard.evaluation.metrics import classification_metrics


def test_metrics_reconstruct_from_case_rows() -> None:
    labels = np.array([True, True, False, False, False])
    scores = np.array([0.9, 0.2, 0.8, 0.4, 0.1])
    result = classification_metrics(labels, scores, 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 2)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["fpr"] == 1 / 3

