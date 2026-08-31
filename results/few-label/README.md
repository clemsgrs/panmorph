# When does another organ's data still help?

PanMorph asks whether labelled slides from one organ can help train an MSI model for
another organ when local MSI-positive cases are scarce. We studied colon cancer
(`COAD`), stomach cancer (`STAD`), and endometrial cancer (`UCEC`).

## Main findings

- **Stomach data gave the colon model the largest and longest-lasting boost.** With 40
  local colon-positive cases, adding the stomach cohort increased AUC from 0.751 to
  0.804: a gain of 0.054 [0.002, 0.107].
- **Colon data mainly helped the stomach model when very few local positives were
  available.** The AUC gain was 0.121 [0.066, 0.177] with three local positives and
  0.067 [0.014, 0.120] with five. By ten positives, the gain was only 0.014, and the
  interval [−0.031, 0.061] included no improvement.
- **Colon data also gave the endometrial model an early boost.** The gain was clear with
  three and five local positives. The estimate stayed positive through 25, but was less
  certain after five. Stomach data gave a smaller boost that was clear only with three
  local positives.
- **Endometrial data added little to either gastrointestinal model.** This agrees with
  the earlier zero-shot result. Adding UCEC to a useful stomach or colon source could
  also weaken it.

## How the comparison works

For each source→target direction, we trained two models:

- **Other-organ + local:** the complete source-organ cohort plus a set number of local
  MSI-positive cases and local negatives at the target organ's usual prevalence.
- **Local only:** exactly the same local cases, without the source-organ cohort.

Within each draw, every target patient was evaluated once by a model that had not trained
on that patient or hospital. The test patients stayed the same as more local cases were
added; only the training data changed.

With no local labels, stomach→colon reached AUC 0.744 [0.681, 0.805], while
colon→stomach reached 0.760 [0.694, 0.820].

## Performance as local labels are added

Each panel shows one source and target organ. For three or more local positive cases,
both models use the same local cases; the blue model also uses the other-organ cohort.
**Blue above black means the other-organ data improved AUC.** At zero local positives,
only the blue zero-shot result is shown. The bands are 95% intervals.

![AUC with and without other-organ data](few_label_value.png)

The first two panels contain the main comparison:

- **STAD→COAD:** the blue curve remained above the local-only curve through 40 local
  positives. At 40, AUC was 0.804 with stomach data and 0.751 without it.
- **COAD→STAD:** the blue curve was clearly higher with three and five local positives.
  At ten, the curves were close: AUC 0.802 with colon data and 0.788 without it. From 25
  onward, the blue curve was slightly lower, but the intervals still included no
  difference.

The endometrial-target panels show smaller early gains:

- **COAD→UCEC:** colon data increased AUC by 0.056 [0.013, 0.097] with three local
  positives and 0.048 [0.007, 0.086] with five. The estimated gain stayed positive
  through 25 positives, but after five the intervals included no gain.
- **STAD→UCEC:** stomach data increased AUC by 0.044 [0.001, 0.084] with three local
  positives. Later gains were smaller and their intervals included no gain.

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

We split hospitals into five groups. “All available” means that, when one group was used
for testing, the model trained on every eligible local patient from the other four
groups. Patients in the test group were never used for training.

The next figure shows the same comparison as an AUC difference. Points above zero favor
adding the other-organ cohort. Zero-shot is omitted because no local-only model exists
when there are no local labels.

![AUC gain from other-organ data](few_label_lift.png)

## How much local data matched a whole source cohort?

We compared two ways to start a model: use all the data from another organ, or use only
local data. We then asked how many local MSI-positive patients were needed to match the
other-organ model:

1. Train on the complete source cohort with no target-organ labels and measure its AUC.
2. Train local-only models with increasing numbers of local positive cases.
3. Find how many local positives are needed to reach the other-organ model's AUC.

For example, the model trained on all 391 colon cases reached the same stomach AUC as a
stomach-only model trained with about eight stomach-positive cases. The local model also
used the usual proportion of MSI-negative stomach cases.

| Source→target | Complete source cohort | AUC with no local labels | Local positives needed to match it |
|---|---:|---:|---:|
| STAD→COAD | 371 cases (63 MSI+) | 0.744 | 31.9 [6.0, ≥59.2] |
| COAD→STAD | 391 cases (74 MSI+) | 0.760 | 8.0 [4.3, 19.7] |
| UCEC→COAD | 487 cases (155 MSI+) | 0.571 | 1.3 [0.0, 2.7] |
| UCEC→STAD | 487 cases (155 MSI+) | 0.521 | 0.4 [0.0, 2.0] |

Each model used the whole source cohort, including its MSI-negative patients. Dividing
these numbers would therefore not give a trustworthy “one source case equals X local
cases” rate.

This calculation is different from the curves above. Here we compare an other-organ-only
model with a local-only model. The curves ask whether other-organ data still helps after
the same local cases have been added to both models.

## How certain are the results?

Five models tested five different groups of hospitals. Their score scales differed a
little—like one teacher grading out of 10 and another out of 100. We repeated the
analysis using only each model's ordering of patients. Stomach data still helped the
colon model, but the estimated gain changed at some label counts. The overall pattern is
more dependable than any one exact AUC gain.

What these results support:

- Existing stomach data improved colon prediction through 40 local positives in the
  primary analysis.
- Existing colon data helps stomach prediction most when only three to five local
  positives are available.
- Existing colon data improved endometrial prediction with three to five local positives;
  stomach data gave a smaller early boost.
- Endometrial data is a weak source for these two gastrointestinal targets and can weaken
  a useful pooled source.

What they do not show:

- A universal conversion between other-organ and local cases.
- How a new organ, dataset, feature representation, or stronger model will behave.
- That other-organ data should replace local labels when local labels are obtainable.
- Why the predictive signal transfers biologically.

## Reproduce the analysis

```bash
python experiments/run_few_label.py --profile quick  # smoke test; not reportable
python experiments/run_few_label.py --profile full   # complete analysis
python -m pytest
```

The main numerical outputs are:

- `few_label_summaries.csv` — AUC with and without other-organ data, plus their difference;
- `few_label_equivalence.csv` — zero-shot equivalent local positives;
- `manifest.json` — the exact run configuration and data identities.
