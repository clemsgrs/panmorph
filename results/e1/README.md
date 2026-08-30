# Integrated E1 full-profile and exploratory swap result

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

## Exploratory COAD-to-STAD budget swap

The downstream swap held the total assay budget at 50, 100, or 200 cases while
replacing COAD cases with STAD cases from 0% through 100% in ten-percentage-point
steps. Each cell contains twenty nested draws. These are exploratory estimates;
they are not confirmatory evidence and do not support a universal source-value
claim.

### Observed findings

Raw pooled STAD OOF AUC is primary. At budget 50, it was 0.5172
[0.4629, 0.5742] at 0% STAD, reached 0.7740 [0.7392, 0.8080] at 90% STAD, and
was 0.7617 [0.7260, 0.7984] at 100% STAD. At budget 100, the corresponding
0%, 90%, and 100% STAD estimates were 0.5665 [0.5144, 0.6221], 0.8219
[0.7845, 0.8591], and 0.8170 [0.7767, 0.8556]. At budget 200, they were 0.6430
[0.5945, 0.6902], 0.8466 [0.8067, 0.8841], and 0.8513
[0.8099, 0.8912]. Rank AUC diverged from raw AUC only at budget 50 with 0% and
10% STAD, budget 100 with 0% STAD, and budget 200 with 0% STAD; rank remains an
annotation and does not replace the raw-AUC results.

Conditional average COAD-case equivalence was 0.1371
[-0.0471, 0.5412] at budget 50 with 50% STAD, 0.1036
[-0.1185, 0.4972] at budget 100 with 50% STAD, and -0.0254
[-0.3517, 0.3134] at budget 200 with 50% STAD. At budget 200, the point estimate
remained negative with 60%, 70%, 80%, and 90% STAD: -0.0845
[-0.4262, 0.2582], -0.2581 [-0.7214, 1.0611], -0.7295
[-1.3444, 0.5559], and -1.5514 [-2.3773, ≥5.8400], respectively. The
budget-200, 90%-STAD interval's upper endpoint was right-censored at the
registered `all` coordinate. Target-only mixtures (100% STAD) have no source cases, so
source-case equivalence is undefined at every budget. Negative values and the
censoring indicator are retained in `swap_equivalence.csv`.

### Findings versus expectations

- The registered expectation that adding local STAD cases could improve STAD
  discrimination was reflected conditionally at every registered budget: the
  raw AUC with 90% STAD exceeded the raw AUC with 0% STAD at budgets 50, 100,
  and 200. The estimates and intervals above show the magnitude and uncertainty.
- The expectation that COAD cases could retain value under target scarcity was
  not uniform across budgets or mixtures. At 90% STAD, conditional average
  COAD-case equivalence was positive at budgets 50 and 100 but negative at
  budget 200, where its upper interval endpoint was right-censored. This
  contrast is descriptive and exploratory; the qualified estimates are above.

## Validation and audit

The retained suite passed before reportable execution (`49 passed` at execution
revision `6db5bf8`), and the expanded post-review suite passed afterward
(`58 passed` after normalized-bundle validation). The full run executed
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

For the downstream swap, the retained suite passed before execution (`80
passed`). The reportable run produced 385,000 draw rows, 244,860 prediction rows,
660 draw-level AUC rows, 33 summary rows, 10 target-reference rows, 33
equivalence rows, and four PNG/PDF figures. Strict validation passed the schemas,
configured keys and joins, nested-prefix invariants, complete 371-patient STAD
OOF coverage in every draw cell, prevalence-matched composition, reproducibility
of derived results, figure signatures, and complete exploratory manifest status.
