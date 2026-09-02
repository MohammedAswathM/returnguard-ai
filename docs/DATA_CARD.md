# Transaction-Grounded Simulation Data Card

The benchmark uses UCI Online Retail II transaction distributions with disclosed simulated refund
operations and abuse labels. It is a reproducible systems benchmark, not production validation.

UCI Online Retail II (DOI `10.24432/C5CG6D`, CC BY 4.0) represents one UK non-store retailer
from 2009-2011. The validated archive contains 1,067,371 rows across two worksheets. It has no
verified refund-abuse labels, delivery proof, device/address network, refund reasons, or inspection
outcomes. Cancellations are retained as transaction history and never treated as abuse labels.

The transformation aggregates invoice baskets, retains source timing/value/frequency/customer/product
structure, pseudonymizes identifiers, and simulates absent operational fields. Missing source
customer IDs become invoice-scoped guest customers. INR amounts use a configurable benchmark FX
assumption and are not historical currency conversions.

Generated `metadata.json` records field-level provenance, source/config/transformed fingerprints,
seed, split boundaries, development prevalence, and difficult-legitimate support. Scenario names,
labels, and outcomes are evaluation-only and excluded from model features.

The four periods are train, calibration, policy selection, and sealed future test. Final support and
timestamps are visible, while final prevalence, labels, scenarios, verifier truth, and features are
redacted until the full decision bundle freezes. A separate cold-start mapping does not replace the
primary chronological replay.

Known gaps include geography/age mismatch, synthetic labels and operational fields, simplified
product grouping, assumed delivery timing and FX, and no protected-group labels. No result establishes
production effectiveness or India-wide validity.
