# E1 full-profile result

This is the reportable `panmorph.e1.bundle/v2` execution on the frozen PRISM
embeddings. Raw pooled out-of-fold AUC is the primary metric. Rank-normalized AUC
is retained only as the registered sensitivity annotation.

## Observed result

The sole confirmatory cell was single-source **COAD → STAD at k=10**. Mean warm
AUC was 0.8020 [0.7497, 0.8513], mean cold AUC was 0.7878 [0.7532, 0.8206], and
the paired raw-AUC lift was **0.0142 [-0.0313, 0.0614]**. Against 999 coherent
within-COAD label permutations, the registered plus-one empirical p-value was
**p=0.001**. This is the bundle's only confirmatory claim. Its rank-sensitivity
lift was 0.0208 and was not flagged as divergent.

The corresponding COAD → STAD foreign-only model was equivalent to **8.00 local
positive cases [4.31, 19.71]**, or **0.0205 local positives per average COAD
source case [0.0110, 0.0504]**. Neither estimate nor interval was censored.

All other cells below are descriptive, exploratory estimates, not additional
confirmatory claims.

| Source → target | Base | Local-positive equivalence [95% interval] | Per-source-case equivalence [95% interval] | Censoring |
|---|---|---:|---:|---|
| COAD → STAD | single | 8.00 [4.31, 19.71] | 0.0205 [0.0110, 0.0504] | none |
| UCEC → STAD | single | 0.40 [0.00, 1.98] | 0.0008 [0.0000, 0.0041] | none |
| COAD+UCEC → STAD | pooled | 6.22 [3.23, 12.76] | 0.0071 [0.0037, 0.0145] | none |
| STAD → COAD | single | 31.93 [6.04, ≥59.20] | 0.0861 [0.0163, ≥0.1596] | upper interval right-censored at `all` |
| UCEC → COAD | single | 1.32 [0.03, 2.66] | 0.0027 [0.0001, 0.0055] | none |
| STAD+UCEC → COAD | pooled | 2.98 [1.76, 17.77] | 0.0035 [0.0021, 0.0207] | none |
| COAD → UCEC | single | 6.40 [1.75, 17.66] | 0.0164 [0.0045, 0.0452] | none |
| STAD → UCEC | single | 6.61 [1.87, 16.81] | 0.0178 [0.0050, 0.0453] | none |
| COAD+STAD → UCEC | pooled | 4.17 [0.91, 11.55] | 0.0055 [0.0012, 0.0152] | none |

Rank sensitivity diverged from raw pooled OOF AUC by more than 0.01 in 33 of 63
summary cells. Flags occurred at every rung for STAD → COAD, STAD+UCEC → COAD,
UCEC → COAD, and UCEC → STAD; at k=0 and `all` for COAD → STAD; and at k=0 for
COAD+UCEC → STAD, COAD+STAD → UCEC, and STAD → UCEC. These flags do not replace
or re-rank the raw-AUC results.

## Findings versus registered expectations

- The expected GI transfer value was present in the one registered test:
  COAD → STAD at k=10 had positive observed lift and exceeded every permuted-null
  statistic (plus-one p=0.001), although its paired bootstrap interval included
  zero.
- The expected weak UCEC-to-GI transfer was reflected descriptively by low
  foreign-only equivalence: 1.32 positives for UCEC → COAD and 0.40 for
  UCEC → STAD.
- The expectation that pooling a weak source could dilute a strong GI source was
  also visible descriptively: pooled equivalence was 2.98 for STAD+UCEC → COAD
  versus 31.93 for STAD → COAD, and 6.22 for COAD+UCEC → STAD versus 8.00 for
  COAD → STAD. These comparisons were not confirmatory tests.
- Rank divergence was more widespread than a fully calibration-stable result
  would predict, chiefly for COAD-target and UCEC-source cells. Accordingly, the
  raw pooled OOF AUC remains primary and rank results remain annotations only.

## Validation and audit

The retained suite passed before reportable execution (`49 passed` at execution
revision `6db5bf8`), and the expanded post-review suite passed afterward
(`57 passed` after normalized-bundle validation). The full run executed
1,224 unique AUC cells and 509,592 predictions, covering every registered target,
single/pooled base, arm, rung, and draw. It used 2,000 paired label-stratified
bootstraps and 999 coherent permutations. The normalized audit stores 123,796
unique local-selection rows, 1,249 exact cohort-case rows, 15 target-fold rows,
and 12 source-base membership rows; their joins reconstruct the original
3,043,144 warm/cold training-membership rows exactly. All nine warm-zero and
three cold-all Phase-1 anchors passed at
`1e-6`; the maximum observed gap was `1.11e-16`.

Both the bundle validator and an explicit audit of schemas, uniqueness keys,
joins, patient coverage, paired local draws, source-base composition, held-out-site
exclusion, ordered null rows, figures, and complete/reportable manifest status
passed.
