# Issue 5 — E1 paired-cell tracer

- [x] Define the public in-memory tracer seam and synthetic cohort fixture.
- [x] Red/green: sample exactly `k` positives and prevalence-matched negatives outside each held-out site fold, without replacement.
- [x] Red/green: pair warm/cold on the same target draw and held-out patients while using the specified training cohorts.
- [x] Red/green: produce exactly one prediction per target patient across five grouped folds.
- [x] Red/green: compute pooled raw AUC and within-fold average-tie percentile-rank sensitivity AUC, flagging only gaps above 0.01.
- [x] Red/green: emit auditable draw, prediction, and AUC records with order-invariant keyed random streams.
- [x] Run focused and full tests; inspect changes for a smaller/elegant implementation.
- [x] Self-review against `main` using the code-review skill, fix actionable findings, and rerun tests.
- [x] Commit, push `agent/issue-5`, and open a PR containing `Closes #5`.
