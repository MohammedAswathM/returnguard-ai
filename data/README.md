# Data sources and limitations

The benchmark transaction foundation is UCI Online Retail II:

- Source: UCI Machine Learning Repository, DOI `10.24432/C5CG6D`
- License: CC BY 4.0
- Geography/time: one UK non-store retailer, 2009-2011
- Official URL: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- Validated archive SHA-256: `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`
- Validated size: 45,622,418 bytes

Run `make download-uci validate-uci data-uci`. The deterministic transformer removes exact
duplicates, aggregates positive priced invoice lines into orders and items, pseudonymizes source
identifiers, retains cancellations as timestamped transaction events, and maps missing customer IDs
to invoice-scoped guest customers. The configured GBP-to-INR conversion is an explicit benchmark
assumption, not a historical exchange-rate claim.

Invoice cancellations are transaction events and never abuse labels. Delivery, payment operations,
refund reasons, requested amounts, devices, addresses, evidence, outcomes, verification fields, and
abuse labels are simulated. `metadata.json` classifies every exported field as source, derived,
simulated, simulated outcome, simulated label, or generator metadata.

The source represents one UK non-store retailer from 2009-2011. It does not validate effectiveness
for modern Indian merchants. Raw archives, sealed truth, and generated artifacts are excluded from Git.
