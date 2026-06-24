# Data inventory & statistics

All numbers verified against files on disk on 2026-06-23. Paths are under
`/data/pathology/projects/clement/mutation-prediction/` unless noted.

## Biomarker labels per cohort

### MSI (`csvs/tcga-<cohort>/dx+msi.csv`, column `msi_high`)

| Organ | Code | MSI+ / n | Prevalence | Status |
|---|---|---|---|---|
| Endometrium | UCEC | 155 / 487 | 32% | **core** |
| Colon | COAD | 74 / 391 | 19% | **core** |
| Stomach | STAD | 63 / 371 | 17% | **core** |
| Bladder | BLCA | 3 / 383 | ~1% | dead (too few positives) |
| Prostate | PRAD | 3 / 398 | ~1% | dead (too few positives) |

→ **Three usable MSI organs** (UCEC, COAD, STAD) → a real 3-way transfer matrix.
PRAD-MSI and BLCA-MSI are unusable as supervised targets; PRAD-MSI may only ever
serve as an *exploratory* zero-shot target in phase 2.

### TP53 (`csvs/tcga-<cohort>/dx+tp53.csv`, column `TP53`)

| Organ | Code | TP53+ / n | Prevalence |
|---|---|---|---|
| Breast | BRCA | 338 / 963 | 35% |
| Bladder | BLCA | 187 / 381 | 49% |
| Prostate | PRAD | 45 / 397 | 11% |

## PRISM feature inventory

PRISM slide-level embeddings: **1280-dim, one vector per `case_id`** (patient-level;
all of a patient's slides already aggregated into a single `.pt`). Stored under
`features/prism/<hash>/features/<case_id>.pt`.

| Cohort | Feature dir hash | # `.pt` | Biomarker(s) ready |
|---|---|---|---|
| COAD | `lxbzb8rd` | 391 | MSI |
| UCEC | `kooqa1ym` | 487 | MSI |
| STAD | `oowdp902` | 371 | MSI |
| PRAD | `or1knvez` | 398 | MSI (dead) + TP53 |

Data-unit integrity (verified): 0 duplicate `case_id`s, 100% of labeled cases matched
to a feature file, feature shape `(1280,)` in all three MSI cohorts. No same-patient
slide leakage is possible because aggregation to one vector per patient is already done.

> ⚠️ **TP53 atlas is blocked.** Only the four PRISM dirs above exist. **BRCA and BLCA
> have no PRISM features extracted** (`features/prism/` contains only the four hashes
> above; the 963-/381-sized dirs under `slide2vec-output/` are *not* confirmed PRISM
> slide-level features). The TP53 transfer atlas (PRAD/BRCA/BLCA) cannot run until
> BRCA + BLCA PRISM features are extracted.

## Tissue-source-site (TSS) structure — the key confound fact

TSS code = the 2-char block in `TCGA-XX-YYYY` (the `XX`). It identifies the
contributing center.

**The three MSI cohorts share ZERO tissue-source-sites.** COAD (24 sites), UCEC
(29 sites), STAD (22 sites); pairwise intersection = 0; no site shared by even a pair.
This is why the zero-shot gate is *structurally* immune to the within-cohort site
shortcut: a source-trained model that keyed on site signatures has no in-distribution
sites to fire on in the target organ. See [experimental-design.md](experimental-design.md).

### Positives are well-dispersed across sites (matters for leave-site-out CV)

Within each MSI cohort, no single site monopolizes the positives and per-site
prevalence sits in a tight band close to the cohort mean:

- **COAD** (74 pos / 24 sites): largest site `AA` = 122 cases (31% of cohort) but
  holds only 24% of positives at 15% prevalence. Per-site prevalence mostly 13–30%.
- **STAD** (63 pos / 22 sites): largest site `BR` = 113 cases holds 32% of positives
  at 18% prevalence. Per-site prevalence mostly 13–27%.
- **UCEC** (155 pos / 29 sites): positives spread broadly; no dominating site.

**Consequence:** label-blind `GroupKFold` (which only equalizes *case counts* per
fold) ends up balancing *positives* per fold too, because positives are diluted
roughly uniformly across many sites. See [methods-notes.md](methods-notes.md) for the
full explanation and the (counterintuitive) finding that `StratifiedGroupKFold` does
*worse* here.

## Site decodability (confound diagnostic)

TCGA tissue-source-site is predictable from PRISM embeddings far above chance —
the reason the site shortcut is a credible confound (which the gate then defeats
structurally; see [experimental-design.md](experimental-design.md)). Multinomial
logistic probe, 5-fold stratified, sites with ≥ 8 cases, balanced accuracy:

| Organ | site bal-acc | chance (1/#sites) | sites≥8 |
|---|---|---|---|
| STAD | 0.734 ± 0.057 | 0.125 | 8 |
| COAD | 0.606 ± 0.059 | 0.077 | 13 |
| UCEC | 0.582 ± 0.014 | 0.071 | 14 |

Reproduce: `python experiments/run_site_probe.py` (uses `panmorph.site.site_decodability`).
This is a *diagnostic only* — not part of the gate decision.

## Reproduce these numbers

```bash
cd /data/pathology/projects/clement/mutation-prediction
# label counts:        read csvs/tcga-*/dx+{msi,tp53}.csv, value_counts on msi_high / TP53
# TSS disjointness:    case_id.split('-')[1] per cohort, set intersections
# feature integrity:   torch.load(features/prism/<hash>/features/<case_id>.pt).reshape(-1).shape
```
