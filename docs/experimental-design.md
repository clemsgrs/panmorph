# Experimental design — the go/no-go gate

Status: design agreed in review (2026-06-23); the gate has since been **run clean and
PASSED** (n_perm=1000, n_boot=2000 — see [results.md](results.md)). Phase-2 design
(decisions 11–25) is settled but not yet implemented.

## Framing: this is a gate, not yet a paper

The immediate deliverable is a **go/no-go decision**, not a transfer atlas or a
method. Question:

> Does *genuine, confound-free* cross-organ transfer of MSI signal exist at all?

A clean "yes, on 3 MSI organs, surviving both confound controls" earns the right to
build the atlas / disentanglement method. A "no" kills the cross-organ story cheaply.
Atlas and method paths are **deferred** until the gate passes.

## The null hypothesis the gate must kill

The original Stage-1 result (prostate TP53 test AUC 0.72 → 0.75 after adding other
organs) is confounded: the enriched run also grew training positives ~24 → ~549
(23×, mostly non-prostate). Three explanations compete:

1. **Sample-size confound** — more training data, regardless of biology.
2. **Site-shortcut confound** — model learns TCGA site/center signatures correlated
   with label, not biomarker morphology.
3. **Genuine cross-organ morphological transfer** — the only interesting one.

**Null to reject:** *"Off-diagonal cross-organ MSI AUC is explained by sample size
and/or site signatures, not shared morphology."*

## Primary gate test: zero-shot cross-organ transfer

Train an MSI classifier on **all of the source organ(s)**, test on **all of a
never-seen target organ**. The target contributes **zero** training cases.

- Kills the **sample-size confound by construction** (target N_train = 0, so "more
  data" cannot explain an above-chance target AUC).
- Kills the **site-shortcut confound by construction** because the MSI cohorts share
  zero TSS (see [data.md](data.md)); a site-keyed model has nothing to fire on in the
  target organ. **We rely on structural site-disjointness, not embedding
  harmonization.** (Agreed: no ComBat arm for the gate.)

Enrichment (target + sources vs target-only) is **retired as the headline** — it can
never separate explanation (1) from (3).

### Pass / fail criteria (pre-registered before running)

Biology is asymmetric: COAD and STAD are both GI / Lynch-dMMR-driven → MSI morphology
(immune infiltration, mucinous/medullary patterns) should transfer. UCEC is
Lynch-associated but gynecological → its MSI morphology may diverge. Therefore:

- **Anchor the decision on the COAD↔STAD pair.**
- A cell **passes** if its zero-shot AUC has **bootstrap 95% CI lower bound > 0.60**
  *and* is significant against the permutation null (below).
- **Strong pass:** both COAD↔STAD directions pass → transfer is real.
- **Partial / interesting pass:** GI↔GI passes but UCEC transfer is weak/asymmetric
  → transfer is real but organ-dependent — *that becomes the atlas thesis*.
- **Fail:** COAD↔STAD does not clear the bar → kill the cross-organ story.

### Mandatory negative control: within-source label permutation null

Shuffle MSI labels **within the source organ** (preserves prevalence + feature
distribution, destroys feature↔label link), retrain, run the identical zero-shot
eval onto the real-labeled target. Repeat ~100–1000×.

- Gives a per-cell null distribution of zero-shot AUCs.
- **Formal significance test:** real zero-shot AUC must exceed the permutation null
  at empirical **p < 0.05**. This subsumes the bare CI check and catches pipeline
  artifacts / leakage (a bug would inflate the shuffled runs too).
- Cheap: logistic regression on ~400 cases × 1000 fits = seconds-to-minutes.

### Within-organ reference ceiling

The matrix diagonal = "how well can we do *within* an organ" = the ceiling that
zero-shot is judged against. **Use the leave-site-out (honest) AUC as the official
ceiling**, so transfer is compared apples-to-apples (both free of the within-cohort
site shortcut). Random-CV AUC is shown only as the *site-inflated* contrast.

One teaching axis per organ: `random-CV → leave-site-out → zero-shot`.
Known site-out vs random-CV gaps (from prelim scripts): STAD +0.02, COAD +0.05,
UCEC −0.01. See [methods-notes.md](methods-notes.md) for the site-out CV protocol.

## Model

- **Primary (now): sklearn logistic regression**, fixed hyperparameters
  (`C=1.0`, `class_weight="balanced"`, `StandardScaler` fit on train only),
  **no tuning** → minimal leakage surface, low variance, cleanest interpretation
  (a *linear* boundary in frozen PRISM space transferring across organs is strong,
  parsimonious evidence). No nested CV needed since nothing is tuned.
- **Follow-up ladder (after gate passes):** torch linear probe, then MLP, as real
  `nn.Module` + train loop (also the migration path toward the disentanglement model).
  Demoted to secondary robustness rows, not co-primary. The ladder continues to a
  **frozen tile encoder + trainable MIL** aggregator — *not* a trainable encoder — as
  an aggregation-capacity control. Full phase-2 model framing (the capacity ladder and
  the shared-subspace method) lives in [research-questions.md](research-questions.md).

## Minimal experiment set for the gate

1. **Zero-shot matrix** — single-source + combined-source, every organ as target once;
   pooled target AUC + patient-level bootstrap 95% CI per cell.
2. **Permutation null** — within-source label shuffle, ~100–1000×, per cell; empirical p.
3. **Within-organ diagonal** — leave-site-out pooled OOF AUC (ceiling) + random-CV (contrast).

That's it. No budget-matched, no few-shot, no TP53 in the gate.

## Deferred to phase 2 (conditional on gate passing)

- **Budget-matched swap** — hold training-set size fixed, swap target cases for source
  cases; "is a foreign case worth a local one?" A second, independent sample-size
  control. **Distinct from the add-on value curve (E1):** the swap holds total `N` fixed,
  the add-on holds source fixed and adds target — see [research-questions.md](research-questions.md).
- **Value — the primary phase-2 RQ** ("how many local positives is a foreign organ
  worth?"), built as a **single measurement engine E1** (subsampling on label-rich organs
  against a real ceiling; its lowest-`k` rungs ≈ 2–3 positives *are* the scarce-target
  regime). A separate PRAD *demonstration* (E2) was dropped as unmeasurable at n=3
  (decision 22). Full design — pair matrix, warm-start bases, injection protocol, anchors,
  prevalence-matched draws, substrate — in [research-questions.md](research-questions.md).
- **TP53 atlas** — blocked on BRCA + BLCA PRISM feature extraction (see [data.md](data.md)).
- **Organ-aware / disentanglement models** — shared vs organ-specific representation,
  only after simple baselines establish the phenomenon. Now scoped in
  [research-questions.md](research-questions.md): a **shared-subspace probe**, demoted to
  a **contingent** experiment (runs only if E1's pooled-base arm shows negative transfer;
  win condition = recover the pooled-base curve to ≈ best single-source base; UCEC-rescue
  a bonus), plus the **PRISM-vs-MIL capacity ladder** (head capacity → aggregation
  capacity; MLP-head rung runs first) as the primary representation/capacity probe. **The
  MIL (aggregation-capacity) rung is blocked on tile-level feature extraction** — only
  1280-d slide-level PRISM exists today (see [data.md](data.md); decision 24); the
  MLP-head rung runs on existing slide vectors and is not blocked.

The research questions these license — the cross-organ **transfer map**, the few-shot
**value** question, and the **mechanism** of organ-dependence — are framed in
[research-questions.md](research-questions.md).

## Decisions log

| # | Decision | Outcome |
|---|---|---|
| 1 | Frame as go/no-go gate, not atlas/method | **Accepted** |
| 2 | Primary gate = above-chance *zero-shot* AUC; retire enrichment as headline | **Accepted** |
| 3 | Pass = CI-lower > 0.60; anchor on COAD↔STAD; UCEC = swing case | **Accepted** |
| 4 | Site control = structural TSS-disjointness, no embedding harmonization | **Accepted** |
| 5 | Mandatory within-source label-permutation null as the formal significance test | **Accepted** |
| 6 | Model: sklearn logreg first (fixed HPs), torch LP/MLP as follow-up ladder | **Accepted** |
| 7 | Defer budget-matched + few-shot to phase 2 | **Accepted** |
| 8 | Within-organ ceiling = leave-site-out AUC; random-CV shown only as inflated contrast | **Accepted** |
| 9 | Site-out splitter = GroupKFold(5) + pooled OOF + train-positive guard (not StratifiedGroupKFold; not averaged per-fold AUC) | **Accepted** |
| 10 | Reporting = confirmatory (COAD↔STAD, gated on CI-lower>0.60 AND perm p<0.05) vs exploratory (UCEC dirs + combined-source, numbers shown, not gated); no global multiple-comparison correction | **Accepted** |
| 11 | Phase-2 model = **frozen tile encoder + trainable MIL** (not a trainable encoder); MIL scoped as an aggregation-capacity control, not the method headline | **Accepted** |
| 12 | Capacity ladder order = linear head → **MLP head first** (settles head capacity, cheap) → MIL (settles aggregation capacity); MLP-first makes any MIL gain attributable to aggregation | **Accepted** |
| 13 | First phase-2 method experiment = **linear shared-subspace probe on frozen PRISM**, subspace fit on source organs only, zero-shot target, `k`-swept, must beat pooled-logreg | **Accepted → amended by 21** (probe demoted to contingent; no longer "first") |
| 14 | Primary phase-2 RQ = **value** ("how many local positives is a foreign organ worth?"); **mechanism demoted** to aspirational, explicitly **blocked on adding organs** (n≈1 transferring pair today) | **Accepted** |
| 15 | Value RQ = **two engines**: E1 subsampling *measurement* on label-rich organs (real leave-site-out ceiling; test held fixed/full, only training positives thinned) + E2 PRAD *demonstration* (ceiling-less, credibility borrowed from E1) | **Accepted → amended by 22** (E2 dropped; value carried by E1 alone, low-`k` rungs as the scarce regime) |
| 16 | E1 computed over the **full pair matrix** (low-`k` lift = free graded transfer readout); **two warm-start bases** (pooled = strong-general-model base; single-source = decomposition); injection = **pooling primary**, shrinkage-to-source optional, freeze-vs-fine-tune deferred to MIL | **Accepted** |
| 17 | Distinguish **add-on value** (source fixed, add `k` target — E1) from **budget-matched swap** (total `N` fixed, trade source↔target — decision 7); report as distinct questions | **Accepted** |
| 18 | **Substrate decoupled by question**: value measurement = parameter-light / fixed-HP / CV-not-tune-split; capacity ladder (linear→MLP→MIL) = trained-NN + tune-split checkpoint selection, **label-rich regime only** | **Accepted** |
| 19 | E1 built inside a **shared site-grouped OOF scaffold**: subsample `k` *within* each fold's training partition (never delete cases from the organ) so the pooled test set is the full organ at every `k`. Ceiling = `cold-start@k=all`, zero-shot = `warm-start@k=0`, all on the **identical site-clean test set**. Budget-matched swap kept **separate** (dual control), not folded into E1 | **Accepted** |
| 20 | E1 significance = **superiority/lift test** (`Δ=AUC_warm−AUC_cold`, paired on shared OOF, same drawn `k` both arms) as the formal gate, with a **within-source label-permutation null** (draws-matched; controls the sample-size confound — *not* a prediction-swap permutation, which ignores draw variance and readmits the confound); substitution rate `k*` reported as **estimation** (CI), not a non-inferiority test; confirmatory cell = COAD→STAD at one pre-registered `k` (decision-10 split) | **Accepted** |
| 21 | Shared-subspace probe **demoted from pillar to contingent** (amends 13): runs **iff E1's pooled-base arm shows negative transfer**; win condition = recover pooled-base curve to ≈ best single-source base; UCEC-rescue kept as bonus only; capacity ladder (decision 12) is the primary "is UCEC a capacity wall" test, not the probe | **Accepted** |
| 22 | **E2 (PRAD demonstration) dropped** (amends 15): 3 MSI positives give no test set on which classifier quality is measurable (AUC CI ≈[0.3,1.0]; one label flip dominates) → "good classifier" unfalsifiable. The measurable scarce-target claim is **E1's lowest-`k` rungs** (≈2–3 positives on STAD/COAD, real ceiling). PRAD not retained even as an anecdote. Also fixes E1's `k` semantics: `k` counts **positives, prevalence-matched** (draw carries negatives at natural rate), since an MSI label costs one assay per case | **Accepted** |
| 23 | **E1 pre-registration tightening** (sharpens 19, 20, 22). (a) `k` is an **absolute count of added target positives** drawn prevalence-matched — grid `k ∈ {0, 3, 5, 10, 25, 50, all}`, *not* percentages (the `%`-vs-count conflation was ambiguous; STAD `3%`≈2 ≠ 3). `all` = every available **non-test-fold** target positive (≈80% of the organ; this *is* `cold-start@k=all`, the ceiling). Negatives per draw = `round(k·(1−p)/p)` at organ prevalence `p`. (b) **Primary draw statistic** = mean Δ over a **fixed set of `D` draw seeds**; each permutation recomputes that *same* mean over the *same* seeds — draws are matched, **not** treated as independent evidence. (c) **Pooled-source null** shuffles labels **within each source organ** (preserves per-organ prevalence + source composition), not globally across the concatenated source. (d) **Pre-registered confirmatory rung** = COAD→STAD, pooled base, **`k=10` positives** — principled by a *realistic local annotation budget* of ≈50 assayed cases (the cost unit is the assayed case, decision 22), which at MSI prevalence ≈17–20% yields ≈10 positives. On the confirmatory target (STAD, 17%) 50 cases ≈ 8–9 pos, so `k=10` (≈59 cases) is the nearest grid rung. Chosen for plausibility, **not** to maximize the lift; lower rungs stay on the curve as exploratory | **Accepted** |
| 24 | **MIL capacity rung depends on tile-level features not yet confirmed available** (qualifies 11): `data.md` verifies only **1280-d slide-level** PRISM (one vector/case); trainable MIL needs frozen-encoder **tile bags**, whose extraction is *not* confirmed. MIL is therefore **blocked pending tile-feature extraction**; the **MLP-head rung is unblocked** (runs on the existing slide vectors) and proceeds first regardless | **Accepted** |
| 25 | **Pooled-OOF metric locked, no post-hoc switch** (sharpens 8, 9): **raw pooled OOF AUC is the primary metric unconditionally**. Rank/quantile-normalized-within-fold pooling is a **pre-specified sensitivity analysis reported alongside** every primary number — *not* a conditional switch triggered by observing (label-informed) divergence. Flag if the two differ by > 0.01 AUC, but the raw number stays primary | **Accepted** |
