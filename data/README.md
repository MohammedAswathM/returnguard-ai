# Data sources and limitations

The default Phase 1 command creates a deterministic synthetic fixture. It does not download UCI
data and its results are not final or production evidence.

The real-base path is prepared around UCI Online Retail II:

- Source: UCI Machine Learning Repository, DOI `10.24432/C5CG6D`
- License: CC BY 4.0
- Geography/time: one UK non-store retailer, 2009-2011
- Official download: `python scripts/download_uci.py`

To use it, set `source_mode: uci` and `uci_archive: data/raw/online_retail_ii.zip` in
`configs/data.yaml`. The deterministic transformer aggregates non-cancellation invoice lines into
orders, pseudonymizes source identifiers, records cancellation event counts as metadata, and then
simulates the absent operational fields and labels. The configured GBP-to-INR conversion is an
explicit benchmark assumption, not a historical exchange-rate claim.

Invoice cancellations are transaction events and must never be used as abuse labels. Refund abuse,
delivery, evidence, devices, addresses, outcomes, and verification fields are simulated. Raw source
archives and generated artifacts are excluded from Git.
