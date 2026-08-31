# panmorph

PanMorph tests whether an MSI classifier trained on one organ can help another organ when
local MSI-positive cases are scarce.

MSI status helps determine eligibility for immunotherapy. It is normally measured with a
laboratory assay; predicting it from routine H&E slides could provide a cheaper screening
tool. The difficulty is that some organs have too few MSI-positive cases to train a
reliable local model.

## Zero-shot transfer

`COAD` is colon cancer, `STAD` is stomach cancer, and `UCEC` is endometrial cancer. Rows
show the training organ; columns show the test organ. Diagonal values are within-organ
hospital-held-out benchmarks.

| Trained on ↓ · Tested on → | Colon (COAD) | Stomach (STAD) | Endometrium (UCEC) |
|---|:---:|:---:|:---:|
| **Colon (COAD)** | _(0.77)_ | **0.76** | 0.59 |
| **Stomach (STAD)** | **0.74** | _(0.86)_ | 0.59 |
| **Endometrium (UCEC)** | 0.57 | 0.52 | _(0.76)_ |

The colon↔stomach results passed the pre-specified uncertainty and shuffled-label checks.
The other cross-organ directions did not. Full values are in
[`results/gate_results.csv`](results/gate_results.csv).

## Few-label transfer

We then added small numbers of labelled patients from the target organ:

- Stomach data still improved the colon model with 40 local MSI-positive cases: AUC was
  0.804 with stomach data and 0.751 without it.
- Colon data helped the stomach model mainly with three to five local positives. It also
  helped the endometrial model at those small sample sizes.
- Endometrial data added little to the colon or stomach models.

See the [full few-label results](results/few-label/README.md) for the figures, uncertainty
intervals, and results at every sample size.

## Data and evaluation

| Organ | Cohort | MSI-positive / total |
|---|---|---:|
| Colon | COAD | 74 / 391 |
| Stomach | STAD | 63 / 371 |
| Endometrium | UCEC | 155 / 487 |

The committed label tables are under `data/`. Frozen 1,280-dimensional PRISM slide
features are stored outside the repository because of their size.

All experiments use a fixed logistic classifier. Hospitals are held out during
evaluation: every target patient is predicted by a model that did not train on that
patient or hospital. The three cohorts also share no tissue-source sites. These choices
reduce the risk that hospital signatures or extra training rows are mistaken for
cross-organ signal.

## Run

```bash
python experiments/run_gate.py                       # zero-shot transfer matrix
python experiments/run_few_label.py --profile quick  # fast smoke test; not reportable
python experiments/run_few_label.py --profile full   # complete few-label analysis
python experiments/run_site_probe.py                 # hospital-site diagnostic
python -m pytest                                      # deterministic test suite
```

The complete few-label run is resumable and writes a validated result bundle to
`results/few-label/`. Its manifest records the data, model, split, seeds, and inference
settings needed to identify the run.

## Repository layout

- `data/` — cohort labels and paths to slide data;
- `src/panmorph/` — data loading, hospital-held-out evaluation, models, and statistics;
- `experiments/` — runnable experiment entry points;
- `results/` — committed result tables, plots, and run manifests;
- `tests/` — deterministic synthetic tests;
- `bib/` — referenced papers.

The current conclusions are specific to TCGA, frozen PRISM features, and the fixed linear
classifier. They do not establish performance for new organs or stronger models.
