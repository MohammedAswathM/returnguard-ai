# Phase 1 Data Card

The default benchmark is a deterministic, fully synthetic fixture designed to exercise causal
infrastructure. All abuse labels, scenarios, evidence, device/address relationships, delivery data,
and outcomes are simulated. Results must be described as a held-out simulation benchmark.

The intended real foundation is UCI Online Retail II (DOI `10.24432/C5CG6D`, CC BY 4.0), representing
one UK non-store retailer from 2009-2011. It has no verified refund-abuse labels and is not evidence
for India-wide or production effectiveness. Cancellations are not fraud labels.

Generated `metadata.json` records the seed, full generator configuration, split support/prevalence,
source mode, disclosures, and SHA-256 fingerprints. Scenario names and outcomes are evaluation-only.

