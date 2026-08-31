# When does another organ's data still help?

PanMorph asks whether labelled slides from one organ can help train an MSI model for
another organ when local MSI-positive cases are scarce. We studied colon cancer
(`COAD`), stomach cancer (`STAD`), and endometrial cancer (`UCEC`).

## Main findings

- **Stomach data gave the colon model the largest and longest-lasting boost.** With 40
  local colon-positive cases, adding the stomach cohort increased AUC from 0.751 to
  0.804: a gain of 0.054 [0.002, 0.107]. A score-normalization check changed the size of
  this gain, so the direction is more reliable than the exact number.
- **Colon data mainly helped the stomach model when very few local positives were
  available.** The AUC gain was 0.121 [0.066, 0.177] with three local positives and
  0.067 [0.014, 0.120] with five. By ten positives, the gain was only 0.014, and the
  interval [−0.031, 0.061] included no improvement.
- **Endometrial data added little to either gastrointestinal model.** This agrees with
  the earlier zero-shot result. Adding UCEC to a useful stomach or colon source could
  also weaken it.

These results describe these cohorts, frozen PRISM slide features, and a fixed logistic
classifier. They are not clinical performance estimates.

## How the comparison works

For each source→target direction, we trained two models:

- **Other-organ + local:** the complete source-organ cohort plus `k` local
  MSI-positive cases and local negatives at the target organ's usual prevalence.
- **Local only:** exactly the same local cases, without the source-organ cohort.

Within each draw, every target patient was evaluated once by a model that had not trained
on that patient or hospital. The test patients stayed the same as `k` increased; only the
training data changed.

AUC measures how well the model ranks MSI-positive patients above MSI-negative patients.
An AUC of 0.5 is chance; 1.0 is perfect. With no local labels, stomach→colon reached AUC
0.744 [0.681, 0.805], while colon→stomach reached 0.760 [0.694, 0.820].

## Performance as local labels are added

Each panel shows one source and target organ. For three or more local positive cases,
both models use the same local cases; the blue model also uses the other-organ cohort.
**Blue above black means the other-organ data improved AUC.** At zero local positives,
only the blue zero-shot result is shown. The bands are 95% intervals.

![AUC with and without other-organ data](e1_value.png)

The first two panels contain the main comparison:

- **STAD→COAD:** the blue curve remained above the local-only curve through 40 local
  positives. At 40, AUC was 0.804 with stomach data and 0.751 without it.
- **COAD→STAD:** the blue curve was clearly higher with three and five local positives.
  At ten, the curves were close: AUC 0.802 with colon data and 0.788 without it. From 25
  onward, the blue curve was slightly lower, but the intervals still included no
  difference.

The table reports every local-data level. Positive values favor adding the other-organ
cohort; an interval spanning zero means the data do not rule out no improvement.

| Local MSI-positive cases | STAD→COAD AUC gain | COAD→STAD AUC gain |
|---:|---:|---:|
| 3 | +0.110 [0.057, 0.158] | +0.121 [0.066, 0.177] |
| 5 | +0.099 [0.048, 0.145] | +0.067 [0.014, 0.120] |
| 10 | +0.067 [0.022, 0.111] | +0.014 [−0.031, 0.061] |
| 25 | +0.062 [0.014, 0.112] | −0.017 [−0.056, 0.021] |
| 40 | +0.054 [0.002, 0.107] | −0.022 [−0.061, 0.018] |
| All available | +0.040 [−0.013, 0.103] | −0.024 [−0.068, 0.020] |

“All available” uses every eligible local training case outside the hospital being
tested.

The next figure shows the same comparison as an AUC difference. Points above zero favor
adding the other-organ cohort. Zero-shot is omitted because no local-only model exists
when there are no local labels.

![AUC gain from other-organ data](e1_lift.png)

## Zero-shot equivalent local data

We also summarized the starting value of each complete source cohort:

1. Train on the complete source cohort with no target-organ labels and measure its AUC.
2. Follow the target-only learning curve as local positives are added.
3. Find where the target-only curve reaches the source-only AUC.

For example, the model trained on all 391 colon cases reached the same stomach AUC as a
stomach-only model trained with about eight stomach-positive cases, plus prevalence-matched
negative cases.

| Source→target | Complete source cohort | Zero-shot AUC | Equivalent target positives |
|---|---:|---:|---:|
| STAD→COAD | 371 cases (63 MSI+) | 0.744 | 31.9 [6.0, ≥59.2] |
| COAD→STAD | 391 cases (74 MSI+) | 0.760 | 8.0 [4.3, 19.7] |
| UCEC→COAD | 487 cases (155 MSI+) | 0.571 | 1.3 [0.0, 2.7] |
| UCEC→STAD | 487 cases (155 MSI+) | 0.521 | 0.4 [0.0, 2.0] |

These are **complete-cohort starting points**, not per-case exchange rates. They combine
source cohort size, class balance, data quality, organ compatibility, representation, and
model. Source cohort size is not controlled. UCEC was larger than COAD or STAD but had
far less equivalent value, so its weak result is not explained by having fewer cases.

The equivalence number also differs from the AUC-gain curves: equivalence compares a
source-only model with a local-only model, while the curves ask whether the source still
helps after local cases have been added.

## One planned shuffled-label check

Before the full run, one comparison was selected to avoid testing every direction and
every local-data level and then highlighting the best result: COAD→STAD with ten local
STAD-positive cases. Ten positives correspond to assaying roughly 50–60 stomach cases at
the observed MSI prevalence.

We randomly reassigned the COAD MSI labels 999 times while keeping the same COAD patients
and local STAD draws. Correct COAD labels performed better than every shuffle (`p=0.001`).
This shows that the COAD labels contained useful information rather than merely adding
more training rows.

It does **not** establish a meaningful improvement over STAD-only training at ten
positives. That improvement was +0.014 AUC [−0.031, 0.061]. The two results coexist
because randomly labelled COAD data usually harmed the model: real COAD labels were much
better than meaningless labels, while their advantage over using no COAD data remained
small.

The original design named a pooled source for this check; implementation issue
[#7](https://github.com/clemsgrs/panmorph/issues/7) changed it to COAD alone before the
full run without recording why. We therefore describe it as a planned check, not as a
fully pre-specified test. STAD→COAD and the other directions need verification in new
data because they were not selected in advance.

## How certain are the results?

The target predictions come from five hospital-held-out models. Their score scales are
not perfectly identical, so we repeated the analysis after ranking scores within each
held-out fold. The STAD→COAD gain stayed positive, but its size changed. The result is
promising evidence of longer-lasting transfer in that direction, not a precise universal
estimate.

What these results support:

- Existing stomach data improved colon prediction through 40 local positives in the
  primary analysis.
- Existing colon data helps stomach prediction most when only three to five local
  positives are available.
- Endometrial data is a weak source for these two gastrointestinal targets and can weaken
  a useful pooled source.

What they do not show:

- A universal conversion between other-organ and local cases.
- How a new organ, dataset, feature representation, or stronger model will behave.
- That other-organ data should replace local labels when local labels are obtainable.
- Why the predictive signal transfers biologically.

## Reproduce the analysis

```bash
python experiments/run_e1.py --profile quick  # smoke test; not reportable
python experiments/run_e1.py --profile full   # complete analysis
python -m pytest
```

The main numerical outputs are:

- `e1_summaries.csv` — AUC with and without other-organ data, plus their difference;
- `e1_equivalence.csv` — zero-shot equivalent local positives;
- `e1_confirmatory_null.csv` — the 999 shuffled-label results;
- `manifest.json` — the exact run configuration and data identities.
