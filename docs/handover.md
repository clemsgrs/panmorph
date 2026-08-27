# Project handover — status and future work

_Written 2026-08-27 for the project-revival meeting. Audience: the incoming project lead._
_This document follows ASD-STE100 style: short sentences, one idea per sentence._

## What this project is

PanMorph maps cross-organ transfer in histology-based genomic biomarker prediction.
The task: predict MSI status from H&E whole-slide images.
The substrate: frozen PRISM slide-level embeddings (1280-d, one vector per patient).
The core question: when a model trains on one organ, does it work on another organ?

The project does **not** claim that multi-organ training always helps.
Early feedback showed that claim is not reliably true.
Instead, we characterize **when** transfer is positive, negative, or absent.

## Status in one paragraph

Phase 1 is complete and it **passed**.
We pre-registered a go/no-go gate and ran it clean on 2026-06-23.
Cross-organ MSI transfer is real, confound-free, and organ-dependent.
COAD↔STAD transfers in both directions.
UCEC does not transfer in any direction.
Phase 2 is fully designed but not implemented.
No work has happened since 2026-06-24.

## The data

All data lives under `/data/pathology/projects/clement/mutation-prediction/`.
See [data.md](data.md) for the verified inventory.

- Three usable MSI organs: UCEC (155/487 MSI-high), COAD (74/391), STAD (63/371).
- PRAD and BLCA are dead for MSI. Each has only ~3 positives.
- The three cohorts share **zero** tissue-source sites. This makes the site confound testable.
- Features: PRISM embeddings, one `.pt` vector per case. Integrity is verified (no duplicates, 100% label–feature match).
- TP53 work is **blocked**. BRCA and BLCA have no PRISM features extracted yet.

## The experiments we ran

Phase 1 was one pre-registered gate plus one diagnostic.
See [experimental-design.md](experimental-design.md) for the design and the 25-row decisions log.

1. **The gate** (`experiments/run_gate.py`). Three parts:
   - Within-organ ceilings with leave-site-out CV (the honest number).
   - A 3×3 zero-shot transfer matrix with patient-level bootstrap CIs.
   - A within-source label-permutation null (empirical p per cell).
   - Pass rule, fixed in advance: CI lower bound > 0.60 **and** perm p < 0.05.
2. **Site-decodability probe** (`experiments/run_site_probe.py`).
   It shows that TCGA site is predictable from the embeddings (bal-acc 0.58–0.73 vs chance ≈0.1).
   So the site shortcut was a credible threat. The gate had to defeat it, and it did.

The model is deliberately simple: a fixed logistic probe, no tuning.
One negative result is recorded in [methods-notes.md](methods-notes.md): raw-space cosine "corroboration" does not work. Do not put it on a slide.

## The main MSI results

See [results.md](results.md) for the full matrix. The verdict is **STRONG PASS**.

| Cell | Zero-shot AUC [95% CI] | perm p |
|---|---|---|
| COAD → STAD | **0.760** [0.697, 0.819] | 0.001 |
| STAD → COAD | **0.744** [0.680, 0.808] | 0.003 |

- Zero-shot GI↔GI recovers almost the full honest within-organ ceiling (COAD site-out 0.767; STAD 0.858).
- UCEC barely transfers in any direction (all p ≈ 0.10–0.44). UCEC is still individually predictable (0.755).
- Headline: **cross-organ MSI transfer is real but organ-dependent.**
- Prior art (Lee et al. 2025, same organs and labels) got ~0.57 for CRC→STAD with tile CNNs. Our frozen-PRISM probe gets 0.760. Both studies agree UCEC is the odd one out. See [related-work.md](related-work.md).
- Cite the full run (1000 perms), not the quick run. The quick run misread COAD→UCEC as significant.

## Where everything lives

| What | Where |
|---|---|
| Code + docs | this repo, `/data/pathology/projects/clement/code/panmorph` |
| Authoritative docs | `docs/` — index at [README.md](README.md) |
| Library | `src/panmorph/` (data, CV, probe, metrics, site probe) |
| Runnable experiments | `experiments/run_gate.py`, `experiments/run_site_probe.py` |
| Committed results | `results/` (`gate_results.csv`, `full.log`, `site_probe.log`) |
| Raw data | `/data/pathology/projects/clement/mutation-prediction/` |
| Phase-2 build spec | [prd-phase-2.md](prd-phase-2.md), also GitHub issue [panmorph#1](https://github.com/clemsgrs/panmorph/issues/1) |
| Tracker | Linear project "panmorph" |
| Bibliography | `bib/` |

Caution: the old deck `phase-2-overview.html` predates decisions 23–25 and contradicts decision 25.
The `docs/` directory is authoritative. The decks are not.

## Future work — the research questions

The agreed scope is **MSI first, as a complete story**. TP53 follows later.
See [phase-2-plan.md](phase-2-plan.md) for the full plan and [research-questions.md](research-questions.md) for the framing.

**RQ1 — Value (primary, agreed).**
*How many local positives is a foreign organ worth?*
Experiment E1 answers this: a warm-start vs cold-start sweep over k added target positives.
The confirmatory rung is pre-registered: COAD→STAD, pooled base, k=10.
E1's k=0 slice reproduces the transfer map for free.
This is the next thing to build. The PRD (34 user stories) specifies it.

**RQ2 — Representation and capacity (secondary).**
*Does the transfer survive, or improve, under a stronger head?*
Ladder: linear → MLP (unblocked, can run now) → MIL (blocked on tile-level features).
Interesting twist: if MIL cracks UCEC, the GI↔endometrium wall was an aggregation artifact, not biology.

**RQ3 — Mechanism (aspirational, blocked).**
*What governs whether transfer happens?*
With one transferring pair, we have n≈1 of variation. Any mechanism story would be unfounded.
This RQ motivates **cohort expansion**: add MSI-labeled organs to grow the matrix.

**Second biomarker — TP53 (later).**
Blocked on PRISM feature extraction for BRCA and BLCA.
TP53 is a good contrast to MSI: its morphology is more context-dependent.

## Suggested first steps for the new lead

1. Read [phase-2-plan.md](phase-2-plan.md), then [results.md](results.md), then the PRD.
2. Reproduce the gate: `python experiments/run_gate.py --quick` (smoke test).
3. Build E1 from the PRD. Story 31 adds the first test suite.
4. In parallel, request PRISM feature extraction for BRCA and BLCA (unblocks TP53).
5. Scope cohort expansion for RQ3 (which TCGA cohorts have usable MSI labels?).
