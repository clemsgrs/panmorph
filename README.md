# panmorph

**PanMorph: Mapping Cross-Organ Transfer in Histology-Based Genomic Biomarker Prediction.**

## Motivation

MSI status decides immunotherapy eligibility. A lab assay is required today.
Predicting MSI from routine H&E slides would be cheaper and faster.
For many organs, labeled cases are scarce. The hope: borrow training data from label-rich organs.

An early pilot seemed to confirm the hope, but it was confounded.
Adding foreign organs also multiplied the training positives ~23×.
More data helps any model. That proves nothing about organs.
So Phase 1 asked one clean question: **is there genuine shared morphology across organs?**

## Data

Labels are committed in this repo: `data/tcga-<cohort>/dx+msi.csv`, column `msi_high`.
Each row also records `wsi_path` and `mask_path`: the packed slide and its tissue mask
on disk. A new feature extractor (e.g. PRISM2) starts from these two columns.
All paths verified present on 2026-08-27.
Features are too large to commit and stay external, under
`/data/pathology/projects/clement/mutation-prediction/`:
`features/prism/<hash>/features/<case_id>.pt`, shape `(1280,)`, one per patient.

| Organ | Cohort | MSI+ / n | Feature hash |
|---|---|---|---|
| Colon | COAD | 74 / 391 | `lxbzb8rd` |
| Stomach | STAD | 63 / 371 | `oowdp902` |
| Endometrium | UCEC | 155 / 487 | `kooqa1ym` |

Integrity is verified: no duplicate case ids, 100% label–feature match.
The three cohorts share **zero** tissue-source sites. This makes the site confound testable.
PRAD and BLCA are unusable for MSI (~3 positives each).

## Phase 1 — the gate (done, PASSED)

Design, fixed before running:

1. Within-organ ceilings with leave-site-out CV (the honest number).
2. Zero-shot matrix: train on source organ(s), test on a never-seen target. Bootstrap CIs.
3. Within-source label-permutation null (1000×), empirical p per cell.
4. Pass rule: CI lower bound > 0.60 **and** p < 0.05, in both COAD↔STAD directions.

Model: fixed logistic probe on the frozen embeddings. No tuning.
A separate probe confirmed hospitals are decodable from the embeddings
(bal-acc 0.58–0.73 vs chance ≈0.1), so the site shortcut was a real threat.
The zero-overlap cohorts and the permutation null defeat it.

### Results (full run, 2026-06-23; raw output in `results/`)

Verdict: **STRONG PASS.**

Zero-shot AUC. Rows: the organ the model trained on. Columns: the organ it was tested on.

| Trained on ↓ · Tested on → | Colon (COAD) | Stomach (STAD) | Endometrium (UCEC) |
|---|:---:|:---:|:---:|
| **Colon (COAD)** | _(0.77)_ | **0.76 ✓** | 0.59 |
| **Stomach (STAD)** | **0.74 ✓** | _(0.86)_ | 0.59 |
| **Endometrium (UCEC)** | 0.57 | 0.52 | _(0.76)_ |

- **✓** = significant, confound-free transfer. COAD→STAD 0.760 [0.697, 0.819], p = 0.001;
  STAD→COAD 0.744 [0.680, 0.808], p = 0.003.
- Plain cells: no significant transfer (all p ≥ 0.10).
- _(Diagonal)_: the organ tested on itself with leave-site-out CV — the honest ceiling, for reference.
- Full precision: `results/gate_results.csv`.

Takeaways:

- GI↔GI transfer is real and nearly recovers the honest ceilings.
- UCEC does not transfer in any direction, although it is predictable within itself.
  Transfer is organ-dependent. That is a finding, not a failure.
- The closest prior work (Lee et al. 2025, PLoS One, `bib/`) found no usable
  transfer on the same organs with tile CNNs. Foundation-model embeddings changed the answer.

Cautions when citing:

- Cite the full run, not `--quick`. The quick run misread COAD→UCEC as significant.
- Do not use raw-space cosine similarity as mechanistic evidence. We checked; it
  contradicts the (real) transfer result. Any mechanism probe must be pre-registered
  in a shared low-dimensional subspace.

## Phase 2 — the value question (designed, not built)

**RQ1 (primary): how many local positives is a foreign organ worth?**

Why this matters clinically: in some organs, MSI-positive patients are very rare.
Prostate is the extreme case — 3 positives in 398 TCGA cases.
Collecting enough positives for a decent supervised model is close to impossible there.
For such organs, borrowed foreign-organ signal may be the only way to get a model at all.
Prostate is also too label-poor to *evaluate* on, so E1 measures the value where the truth
is known: it simulates scarcity by subsampling the label-rich organs.
The analysis works for any organ mix; prostate is simply the example with the
strongest clinical motivation.

Experiment **E1**: sweep k added target positives (prevalence-matched draws).
Compare warm start (begin from the foreign-organ model) against cold start (local cases only).
The gap between the curves is the value, in local labeled cases saved.
Pre-registered confirmatory cell: **COAD→STAD, pooled base, k=10**
(≈ a realistic 50-case local annotation budget).
Statistic: paired lift Δ(k), averaged over fixed draw seeds; per-organ label-permutation null.
Full specification: GitHub issue [panmorph#1](https://github.com/clemsgrs/panmorph/issues/1).
It opens with a plain-language brief; the precise build spec follows below it.
The first E1 tracer and its synthetic pytest coverage now establish the experiment seam.
The registered matrix runner expands that seam over every single-source and pooled
all-non-target base for COAD, STAD, and UCEC. It writes separate audit, prediction,
and pooled-AUC tables only after its deterministic endpoints reproduce the committed
Phase-1 results within `1e-6`.

**RQ2 (secondary): does a stronger model change the picture?**
Try a newer foundation model (e.g. PRISM2) or a tile encoder with a trainable MIL
aggregator (needs tile-level feature extraction).
If a stronger model makes UCEC transfer, its wall was representational, not biological.

**On the shelf: steering the model toward organ-agnostic morphology.**
In earlier explorations we tried to force cross-organ features directly:

- A domain-adversarial loss (gradient reversal on organ identity) to remove the
  organ signal from the representation.
- A factorized latent space, split into organ-specific and organ-agnostic parts,
  kept apart with an orthogonality constraint.

These are worth reviving if plain training does not deliver the transfer we want.
Two cautions from earlier feedback: adversarial organ removal can hurt, and pure
organ invariance may be too strong — the best model may need both feature types.

Explaining *why* transfer happens is deliberately out of scope for now.
With one transferring pair, there is nothing to generalize from.
It becomes a real question only after more MSI-labeled organs join the matrix.

## Layout

- `data/` — committed MSI label CSVs, one directory per cohort.
- `src/panmorph/` — library (data loading, CV, probe, metrics, E1 tracer, site probe).
- `experiments/` — runnable entry points.
- `results/` — committed gate output (`gate_results.csv`, logs).
- `handover-deck.html` — the project walk-through deck.
- `bib/` — referenced papers.

## Running

```bash
python experiments/run_gate.py           # full gate: 1000 perms, 2000 bootstraps
python experiments/run_gate.py --quick   # smoke test
python experiments/run_site_probe.py     # site-decodability diagnostic
python experiments/run_e1.py             # full registered E1 matrix (20 paired draws)
python -m pytest                         # deterministic synthetic test suite
```
