# Data Card

## Foundation

The benchmark uses UCI Online Retail II under CC BY 4.0. The validated archive contains 1,067,371 transaction rows from December 2009 through December 2011. Archive SHA-256 is `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`.

UCI fields provide invoices, products, quantities, prices, customers, countries, and timestamps. Derived fields include baskets, order values, tenure, frequency, cancellation history, and chronological histories. Refund requests, delivery/payment operations, device/address links, evidence, verifier results, scenarios, costs, and abuse labels are simulated and named accordingly in metadata.

UCI cancellation rows remain transaction events and are never abuse labels.

## Partitions

Four non-overlapping chronological periods contain 4,800 train, 800 calibration, 800 policy-selection, and 1,600 locked future-test requests. Cold-start is a separate challenge mapping and does not alter primary test membership.

Point-in-time computation requires `event_time < decision_time`, matured outcomes, and compute-before-update ordering. Final truth remained physically sealed until a freeze manifest bound the model, calibration, likelihoods, policy, costs, features, source, and transformed data.

## Difficult legitimate cases

The generator includes shared households, loyal high-volume customers, product-defect bursts, carrier incidents, legitimate high-value damage, first-order returns, and missing or inconclusive evidence. Several held-out slices are small and must not be treated as conclusive subgroup evidence.

## Limitations

The original retail history is UK-centric. Currency conversion and all refund-risk semantics are simulated. The dataset does not validate production fraud prevalence, merchant operations, Indian consumer behavior, or realized policy value.

V1 contains original captured payment amounts but no point-in-time prior-refund ledger. Fifty-five final requests exceed original captured payment and are reported as a separate deterministic-integrity subgroup. Requests below original captured amount are not asserted to have verified refundable balance.
