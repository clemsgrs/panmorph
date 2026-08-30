# From “does transfer exist?” to “when is it useful?”

Phase 1 showed that MSI signal transfers in both directions between colon cancer
(**COAD**) and stomach cancer (**STAD**) when the target organ contributes no training
labels. This Phase-2 experiment asks the practical next question:

> Once local labels start arriving, how much does the foreign-organ data still help?

Endometrial cancer (**UCEC**) is included as a weak-transfer comparison. All experiments
use the same fixed logistic model and frozen PRISM features.

This page reports the E1 value experiment and its fixed-budget follow-up. It does not
answer the separate stronger-model/capacity question.

## The answer in one minute

1. **Both GI directions transfer at zero shot, but they do not have equal practical
   value.** The reverse direction, STAD→COAD, retained useful signal for substantially
   longer as local labels were added.
2. **The complete COAD cohort was worth about 8 local STAD-positive cases.** Its
   advantage was clear with 3–5 local positives but largely gone by 10.
3. **The complete STAD cohort was worth about 32 local COAD-positive cases.** Its raw-AUC
   advantage remained positive through 40 local positives. This result is exploratory
   and sensitive to how scores from hospital-held-out folds are put on a common scale.
4. **UCEC did not become a useful GI source.** Worse, adding UCEC to the strong GI source
   diluted its estimated value, especially for COAD.
5. **Foreign cases should not generally replace local cases at equal labeling cost.**
   In the fixed-budget experiment, all-local training was usually better. The one near
   tie occurred at the largest reverse-direction budget and was too uncertain to support
   a sampling rule.

The main new result is therefore not another zero-shot score. It is the finding that
**foreign-data value decays at different rates in the two directions.**

## What E1 adds beyond zero shot

Zero shot is the leftmost point, `k=0`: the model has all foreign labels and no local
labels. E1 then gives both models the same `k` local MSI-positive cases, plus negatives at
the target organ's natural prevalence:

- **Foreign + local:** complete foreign cohort plus those local cases.
- **Local only:** exactly the same local cases, without the foreign cohort.

Their AUC difference tells us what the foreign data added after controlling for the local
training sample. The complete target cohort remains the test set at every point.

| Direction | Phase-1 zero-shot AUC | Added AUC at 3 local positives | Added AUC at 10 local positives | Complete foreign cohort worth |
|---|---:|---:|---:|---:|
| COAD→STAD | 0.760 | +0.121 [0.066, 0.177] | +0.014 [−0.031, 0.061] | 8.0 STAD positives [4.3, 19.7] |
| STAD→COAD | 0.744 | +0.110 [0.057, 0.158] | +0.067 [0.022, 0.111] | 31.9 COAD positives [6.0, ≥59.2] |

![Warm and local-only value curves for every source and target](e1_value.png)

The first two panels are the important comparison:

- **COAD→STAD:** foreign data helped strongly at `k=3` and still helped at `k=5`.
  By `k=10`, foreign+local AUC was 0.802 versus 0.788 for local-only—a small, uncertain
  difference. With more local data, the point estimate became slightly negative.
- **STAD→COAD:** foreign+local remained ahead of local-only through `k=40`; at that point
  the lift was +0.054 [0.002, 0.107]. Even at the all-local endpoint the point estimate
  remained positive, although its interval included zero.

Only COAD→STAD at `k=10` was pre-registered for a confirmatory test. STAD→COAD is a
prominent exploratory finding, not a second confirmed result.

![AUC added by foreign data for every source and target](e1_lift.png)

The shaded `k=0` region is the Phase-1 zero-shot anchor. It is isolated because there is
no trained local-only comparator at zero labels; the reference is chance AUC 0.5. The
points at `k>0` are the Phase-2 answer: what foreign data added beyond the same local
training set.

### Why p=0.001 can coexist with an interval crossing zero

These numbers answer different questions:

- The label-permutation result, `p=0.001`, says correctly paired COAD labels added more
  value than the same number of COAD rows with their labels shuffled.
- The lift interval, −0.031 to +0.061, says the exact AUC gain at 10 local STAD positives
  could be negligible.

So COAD carries real transferable signal, but its remaining performance gain at that
particular local-data budget is small and uncertain.

## What UCEC taught us

UCEC remained a weak source for the two GI targets:

- UCEC alone was worth 0.40 local STAD positives [0.00, 1.98].
- UCEC alone was worth 1.32 local COAD positives [0.03, 2.66].
- COAD+UCEC was worth 6.22 STAD positives, less than COAD alone at 8.00.
- STAD+UCEC was worth 2.98 COAD positives, far less than STAD alone at 31.93.

That last contrast is more informative than another zero-shot score: a weak organ can
dilute a useful source after local labels are introduced. It motivates, but does not yet
validate, methods that try to retain shared signal while ignoring non-sharing organs.

## Fixed labeling budget: add foreign cases or replace local cases?

E1 adds a complete foreign cohort on top of local data. The swap experiment asks a
stricter and different question: if the total labeling budget is fixed at 50, 100, or 200
patients, should any local cases be replaced by foreign cases?

![Fixed-budget composition curves in both GI directions](swap_auc.png)

The dashed line in each panel is the Phase-1 zero-shot AUC from the **complete** foreign
cohort. The 0%-local points are lower because they use only 50, 100, or 200 sampled
foreign cases.

At a 50/50 split, the comparison was:

| Foreign→target | Total cases | Half foreign / half local AUC | All-local AUC |
|---|---:|---:|---:|
| COAD→STAD | 50 | 0.714 [0.682, 0.747] | 0.762 [0.726, 0.798] |
| COAD→STAD | 100 | 0.777 [0.739, 0.811] | 0.817 [0.777, 0.856] |
| COAD→STAD | 200 | 0.815 [0.774, 0.854] | 0.851 [0.810, 0.891] |
| STAD→COAD | 50 | 0.644 [0.603, 0.684] | 0.721 [0.677, 0.763] |
| STAD→COAD | 100 | 0.718 [0.673, 0.760] | 0.756 [0.707, 0.803] |
| STAD→COAD | 200 | 0.762 [0.712, 0.810] | 0.753 [0.696, 0.810] |

All-local training had the higher point estimate in five of the six comparisons. The
exception—STAD→COAD at 200 total cases—was a near tie with heavily overlapping
intervals, not evidence that replacing COAD cases is better.

### There is no stable “one foreign case equals X local cases” rule

At the 50/50 mixture, the estimated local-case value of one foreign case changed sharply
with budget:

| Direction | Budget 50 | Budget 100 | Budget 200 |
|---|---:|---:|---:|
| One COAD case in STAD units | +0.137 [−0.047, +0.541] | +0.104 [−0.118, +0.497] | −0.025 [−0.352, +0.313] |
| One STAD case in COAD units | −0.429 [−0.537, −0.081] | −0.012 [−0.285, +3.500] | +1.800 [−0.172, ≥2.128] |

![Conditional average foreign-case equivalence](swap_equivalence.png)

The wide and censored intervals are the result, not a plotting nuisance. They show that a
single exchange rate would be misleading: value depends on direction, total budget, and
the current source/target mixture.

## Robustness warning

The target is evaluated through five hospital-held-out folds. Scores from those five
models are not always on exactly the same numerical scale. We therefore also ranked
scores within each fold and flagged any result that changed by more than 0.01 AUC.

The STAD→COAD E1 curve was flagged at every rung, and many reverse swap cells were also
flagged. The raw pooled AUC remains the registered primary metric, but the reverse
effect sizes should be read as promising rather than precise.

## What we can and cannot claim

Supported by the registered test:

- COAD labels contain real MSI signal that transfers to STAD when 10 local STAD-positive
  cases are available (`p=0.001`).
- The remaining AUC gain at that operating point was small and uncertain.

Exploratory findings:

- STAD may be substantially more valuable to COAD than COAD is to STAD.
- UCEC can dilute a useful GI source rather than improve it.
- At fixed labeling cost, local cases were usually more useful than foreign replacements.

Not supported:

- A universal foreign-to-local case exchange rate.
- A recommendation to replace local labeling with foreign-organ labeling.
- A claim that a stronger model would preserve or change this transfer map; that
  capacity experiment has not been run.

## Reproducibility

The reportable E1 run contains all 1,224 registered cells, 2,000 bootstrap replicates,
and 999 label permutations for the sole confirmatory cell. The bidirectional swap adds:

- 1,320 direction×budget×mixture×draw AUC cells;
- 502,920 patient predictions;
- 770,000 audited training selections;
- 66 directional summaries and 66 case-equivalence estimates;
- 2,000 bootstrap replicates and no confirmatory permutation test.

Every patient appears exactly once per out-of-fold evaluation. After removing the two
new direction columns, the original COAD→STAD swap rows are numerically and row-for-row
unchanged.

Detailed values are in:

- `e1_summaries.csv` — foreign+local, local-only, and lift estimates;
- `e1_equivalence.csv` — complete-source value in local-positive units;
- `e1_confirmatory_null.csv` — 999 shuffled-label results;
- `swap_summaries.csv` — bidirectional fixed-budget AUC estimates;
- `swap_equivalence.csv` — direction-, budget-, and mixture-specific case values;
- `manifest.json` — configuration and execution identity.
