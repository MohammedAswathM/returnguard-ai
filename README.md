# ReturnGuard AI

Phase 0/1 implements a reproducible causal benchmark and transparent rules/Logistic Regression
baselines for serial false-claim risk before an instant or returnless refund. It is not a generic
fraud dashboard, and all current performance is a held-out simulation benchmark.

## Setup and verification

Python 3.11 or 3.12 is supported.

```bash
python3 -m pip install -e ".[dev]"
make data
make train-baselines
pytest -q
make verify
```

`make data` uses the deterministic fixture in `configs/data.yaml`. To fetch the licensed public
transaction base separately:

```bash
python scripts/download_uci.py --output data/raw/online_retail_ii.zip
```

Set `source_mode: uci` and the downloaded `uci_archive` path in `configs/data.yaml` to use the
deterministic real-base transformer. Fixture results must not be described as UCI-derived or final.
See `data/README.md` and `docs/DATA_CARD.md`.

## Outputs

- `artifacts/data/*.csv`: canonical tables and cold-start mapping
- `artifacts/data/metadata.json`: configuration, split summary, disclosures, fingerprints
- `artifacts/baselines/feature_snapshot.csv`: causal development-period feature rows
- `artifacts/baselines/case_predictions.csv`: reconstructable development-period predictions for both baselines
- `artifacts/baselines/metrics.json`: support, prevalence, confusion counts, precision, recall, F1,
  FPR, and average precision
- `artifacts/baselines/logistic_pipeline.joblib`: locally trained Phase 1 baseline artifact

No dashboard, API, LightGBM, calibration, verifier policy, or Razorpay integration is included yet.
Final-test labels, features, predictions, and prevalence remain unopened until the Phase 2 bundle and
policy are frozen; Phase 1 exports only final support and timestamp boundaries.
