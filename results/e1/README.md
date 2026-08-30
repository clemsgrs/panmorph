# What did foreign-organ training data buy us?

We tested whether MSI labels from one organ can reduce the number of local labels
needed for another organ. The model and frozen PRISM features were kept fixed.

The organs are:

- **COAD** — colon cancer
- **STAD** — stomach cancer
- **UCEC** — endometrial cancer

## Bottom line

1. **Colon data carried real signal for stomach cancer, but the measured performance
   gain was small and uncertain.** With 10 local STAD-positive cases, adding COAD data
   raised AUC from 0.788 to 0.802: a gain of 0.014. The uncertainty interval for that
   gain included no improvement.
2. **The complete COAD dataset was worth roughly 8 local STAD-positive cases.** The
   plausible range was broad: about 4 to 20 cases.
3. **UCEC was of little help for the two gastrointestinal cancers.** Its estimated value
   was only 1.3 COAD-positive cases or 0.4 STAD-positive cases.
4. **Adding a weak source could dilute a strong one.** Pooling UCEC with a useful GI
   source reduced the estimated value in both GI targets.
5. **At a fixed labeling budget, local STAD cases did most of the work.** Replacing a
   small number of STAD cases with COAD cases sometimes gave a slightly higher point
   estimate, but the differences were uncertain and did not support a general rule.

The safe conclusion is: **cross-organ signal exists, but its practical value depends on
the source, target, and amount of local data. There is no universal exchange rate for a
foreign case.**

## How to read the numbers

- **AUC** measures how well the model separates MSI-positive from MSI-negative patients.
  `0.5` is chance; `1.0` is perfect separation.
- Values in brackets are **95% uncertainty intervals**. Wider intervals mean less
  certainty about the size of the effect.
- **Equivalent local positives** asks: “How many local MSI-positive cases would a model
  trained only on the target organ need to match the foreign-only model?”
- `≥` means the upper end could not be measured before reaching all available local data.

Two results that look contradictory answer different questions:

- The permutation result, **p=0.001**, says the COAD labels contain transferable signal;
  none of the 999 shuffled-label runs matched the observed result.
- The lift interval, **-0.031 to 0.061**, says the exact performance benefit at 10 local
  positives is uncertain and could be negligible.

So there is strong evidence of transferable predictive signal, but not strong evidence
that its immediate AUC gain is large.

## Main result: colon to stomach

This was the only pre-registered confirmatory comparison. Both models received the same
local training sample, built around 10 STAD-positive cases plus negatives at the natural
STAD prevalence; the warm model also received the complete COAD dataset.

| Model | Mean AUC | 95% interval |
|---|---:|---:|
| STAD cases only | 0.788 | 0.753 to 0.821 |
| COAD plus the same STAD cases | 0.802 | 0.750 to 0.851 |
| Difference | **+0.014** | **-0.031 to +0.061** |

The complete COAD dataset was estimated to be worth **8.00 local STAD-positive cases**
(4.31 to 19.71). Dividing by the 391 COAD patients gives an average of 0.0205 local
positive cases per COAD patient. That average is useful for accounting, not as a claim
that every COAD patient has equal value.

## What the other organ pairs suggested

These comparisons are exploratory. They are useful patterns, not additional confirmed
findings.

| Foreign training data | Target | Equivalent local positive cases | Plain-language reading |
|---|---|---:|---|
| COAD | STAD | 8.00 [4.31, 19.71] | Useful transfer |
| UCEC | STAD | 0.40 [0.00, 1.98] | Little measurable value |
| COAD + UCEC | STAD | 6.22 [3.23, 12.76] | Less than COAD alone |
| STAD | COAD | 31.93 [6.04, ≥59.20] | Potentially strong, but upper bound unresolved |
| UCEC | COAD | 1.32 [0.03, 2.66] | Little measurable value |
| STAD + UCEC | COAD | 2.98 [1.76, 17.77] | Much less than STAD alone |
| COAD | UCEC | 6.40 [1.75, 17.66] | Modest exploratory value |
| STAD | UCEC | 6.61 [1.87, 16.81] | Modest exploratory value |
| COAD + STAD | UCEC | 4.17 [0.91, 11.55] | Pooling did not improve the estimate |

The clearest descriptive pattern was that UCEC contributed little to COAD or STAD and
reduced the estimated value when pooled with the stronger GI source.

### A robustness warning

For 33 of the 63 E1 summary cells, AUC changed by more than 0.01 when scores were ranked
within each hospital-held-out fold. This means some results depend on how scores from
different folds are put on a common scale. The reported raw AUC remains the primary
metric; the rank analysis is only a warning flag.

## Fixed-budget experiment: should some STAD labels be replaced with COAD labels?

The first experiment added foreign data on top of local data. This second, exploratory
experiment asked a stricter question: with a fixed budget of 50, 100, or 200 labeled
patients, should any of those labels come from COAD instead of STAD?

The table shows three easy-to-interpret mixtures. Every estimate averages 20 repeated
draws and evaluates the complete STAD cohort.

| Total labeled patients | 100% COAD | 90% STAD + 10% COAD | 100% STAD |
|---:|---:|---:|---:|
| 50 | 0.517 [0.463, 0.574] | **0.774 [0.739, 0.808]** | 0.762 [0.726, 0.798] |
| 100 | 0.566 [0.514, 0.622] | **0.822 [0.785, 0.859]** | 0.817 [0.777, 0.856] |
| 200 | 0.643 [0.594, 0.690] | 0.847 [0.807, 0.884] | **0.851 [0.810, 0.891]** |

What this says:

- **COAD-only training was much worse than training with mostly or entirely STAD.**
- With budgets of 50 and 100, the 90%-STAD mixture had a slightly higher point estimate
  than 100% STAD. The intervals overlap heavily, so this is not evidence that replacing
  STAD cases with COAD cases is reliably better.
- With a budget of 200, 100% STAD had the highest of these three point estimates.
- Score-ranking warnings appeared only in mixtures with no or very little STAD: 0% STAD
  at every budget and 10% STAD at budget 50.

### How much was one COAD case worth at a 50/50 mixture?

This calculation expresses the average COAD case in units of local STAD-positive cases.

| Total budget | Estimated value per COAD case | 95% interval |
|---:|---:|---:|
| 50 | +0.137 | -0.047 to +0.541 |
| 100 | +0.104 | -0.118 to +0.497 |
| 200 | -0.025 | -0.352 to +0.313 |

All three intervals include zero. We therefore cannot conclude that an average COAD case
has consistently positive value at a 50/50 mixture.

At budget 200, the point estimate stayed negative from 50% through 90% STAD and became
more negative as COAD became scarcer. Those intervals were wide and included zero; the
90%-STAD upper bound also exceeded the measurable local-data range. This is an unstable,
exploratory pattern—not evidence that COAD cases are universally harmful.

At 100% STAD there are no COAD cases, so “value per COAD case” is undefined.

## What we can and cannot claim

### Supported by the registered test

- COAD labels contain MSI signal that transfers to STAD when 10 local STAD-positive cases
  are available (`p=0.001`).
- The observed AUC gain was small, and its uncertainty interval included zero.

### Exploratory only

- UCEC appears to offer little value for COAD or STAD.
- Pooling UCEC with a useful GI source may dilute that source.
- A small COAD fraction may sometimes match an all-STAD budget, but this was not consistent
  enough to recommend a universal sampling strategy.
- The value of a COAD case changes with total budget and COAD/STAD mixture.

## Validation and detailed artifacts

The final test suite contains 80 deterministic tests. The reportable runs covered:

- all 1,224 registered E1 cells, 2,000 bootstraps, and 999 label permutations;
- all 660 fixed-budget swap cells, 2,000 bootstraps, and no confirmatory permutations;
- every patient exactly once in each out-of-fold evaluation;
- exact Phase-1 anchor reproduction, with a maximum difference of `1.11e-16`;
- schema, key, join, nesting, prevalence, and artifact-integrity checks.

The normalized audit tables can reconstruct every training set without storing millions
of repeated rows. Exact values are available in:

- `e1_summaries.csv` — E1 AUC and lift estimates
- `e1_equivalence.csv` — local-positive equivalence estimates
- `e1_confirmatory_null.csv` — the 999 shuffled-label results
- `swap_summaries.csv` — fixed-budget AUC estimates
- `swap_equivalence.csv` — budget- and mixture-specific COAD-case values
- `manifest.json` — registered configuration and execution identity

All E1 claims outside the single COAD-to-STAD comparison at 10 local positives, and all
fixed-budget swap results, remain exploratory.
