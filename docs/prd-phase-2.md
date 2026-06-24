# Phase-2 PRD — value measurement (E1) + capacity ladder + contingent probe

Published to the issue tracker as [panmorph#1](https://github.com/clemsgrs/panmorph/issues/1)
(`ready-for-agent`). This file is the in-repo copy.

> **Source of truth:** this PRD synthesizes the phase-2 design in `docs/` (decisions
> 11–25 in [experimental-design.md](experimental-design.md), the E1 spec in
> [research-questions.md](research-questions.md) and [phase-2-plan.md](phase-2-plan.md),
> and the CV/significance/imbalance rationale in [methods-notes.md](methods-notes.md)).
> Where this PRD and those docs disagree, the docs win. The go/no-go gate has already
> **PASSED** ([results.md](results.md)); phase 2 builds on it.

## Problem Statement

The gate proved that *confound-free, organ-dependent* cross-organ MSI transfer exists on
frozen PRISM embeddings: COAD↔STAD transfer (zero-shot AUC 0.760 / 0.744, perm p<0.005),
while UCEC transfers in no direction. That is a **binary existence** result. It does not
tell a researcher facing a **new organ with very few MSI labels** the thing they actually
need to decide: *is it worth assembling a foreign-organ training set, and if so, how many
local labels does that foreign model save me?* Today we can only say "transfer happens for
GI↔GI"; we cannot quantify what it is worth, cannot say whether a stronger representation
changes the map, and cannot say whether multi-organ pooling helps or hurts. There is also
no experiment code beyond the gate, no test suite, and several design commitments (exact
`k` semantics, draw-aggregation statistic, the pooled-source null) that are precise in
prose but not yet enforced anywhere.

## Solution

Implement the **phase-2 experiment spine** that converts the binary transfer result into a
**continuous substitution rate** and stress-tests it. Four experiments, sharing one
site-grouped out-of-fold (OOF) scaffold so every number sits on the same site-clean test
set:

1. **E1 — value measurement (load-bearing).** Few-shot value *curves* on label-rich
   organs: simulate scarcity by subsampling the target's training positives down a `k`
   sweep, with a real organ-specific ceiling to check against. The lift of a foreign-base
   *warm-start* over a *cold-start* (from-scratch) model **is** the value. Reports a
   substitution rate `k*` and one pre-registered confirmatory significance test.
2. **Budget-matched swap (secondary dual control).** Hold total training `N` fixed and
   trade source cases for target cases — "at equal labeling cost, is a local case worth
   more than a foreign one?"
3. **Capacity ladder (RQ2).** Is the transfer map an artifact of representation capacity?
   `linear head → MLP head` (head capacity) then `frozen-encoder + trainable MIL`
   (aggregation capacity).
4. **Shared-subspace probe (contingent).** Runs only if E1's pooled multi-organ base shows
   negative transfer; tries to recover the pooled-base curve to ≈ the best single-source
   base by projecting onto a source-only learned shared subspace.

The deliverable is the experiment code, a tidy results schema, and the **first test
suite**, with per-component seams that lock the statistical commitments in place.

## User Stories

1. As a researcher with a label-scarce new organ, I want a curve of model quality vs.
   number of local positives, so that I can decide how many cases to assay before a
   foreign-organ model stops helping.
2. As a researcher, I want the curve's x-axis to count **positives** drawn
   prevalence-matched, so that it reflects the real cost of assaying whole cases rather
   than an unattainable balanced sample.
3. As a researcher, I want a single substitution rate `k*` (with a CI), so that I can say
   "a foreign model is worth ~N local positives" in one number.
4. As a skeptical reviewer, I want the value claim tested against a null that holds source
   data **volume** fixed and randomizes only labels, so that a positive result means
   transferred biology and not just "more training rows."
5. As a skeptical reviewer, I want the warm-start vs cold-start comparison to be paired on
   the **same drawn `k` cases** and the **same full-organ test set**, so that the lift `Δ`
   isolates the value of source data and cancels any shared test-time artifact.
6. As a researcher, I want exactly **one** pre-registered confirmatory test (COAD→STAD,
   pooled base, `k=10` positives), so that I am not implicitly running seven correlated
   per-`k` tests and inflating significance.
7. As a researcher, I want `k=10` justified by a realistic ≈50-case annotation budget at
   MSI prevalence ≈17–20%, so that the headline operating point is clinically plausible
   rather than the `k` that happens to maximize the lift.
8. As a researcher, I want the low-`k` rungs (`k=3–5` positives) reported as exploratory,
   so that the genuinely-scarce few-shot regime is visible without being what the
   confirmatory test is staked on.
9. As a researcher, I want every `k` evaluated on the **full target organ** as the test
   set, so that AUCs at different `k` are comparable (only training composition changes).
10. As a researcher, I want the `k` added cases drawn from sites **disjoint** from the test
    fold, so that warm-start gets no within-organ site shortcut the ceiling lacks.
11. As a researcher, I want `k=0` to reproduce the gate's zero-shot AUC and `k=all` to
    reproduce the cold-start leave-site-out ceiling, so that the scaffold is anchored to
    already-published numbers.
12. As a researcher, I want results averaged over a **fixed set of `D` draw seeds** with
    the null recomputing the same mean over the same seeds, so that draws tighten the
    estimate without being mistaken for independent evidence.
13. As a researcher running the pooled multi-organ base, I want the permutation null to
    shuffle labels **within each source organ**, so that the null preserves each organ's
    prevalence and the source composition the real run uses.
14. As a researcher, I want the full pair matrix computed (not just the headline cell), so
    that the low-`k` lift per pair gives a free graded transfer readout (large for
    COAD→STAD, ≈0 for UCEC→anything).
15. As a researcher, I want two warm-start bases — pooled multi-organ and single-source —
    so that I can separate "is a strong general model worth warm-starting from?" from
    "which foreign organ carries the value?".
16. As a researcher, I want pooling (`refit on source ∪ k target`) as the primary injection
    and shrinkage-to-source as an optional secondary, so that the main result uses the same
    fixed-HP estimator as the gate with the fewest knobs.
17. As a researcher, I want a budget-matched swap reported as a **separate** question from
    E1's add-on curve, so that the one number E1 delivers (foreign value at fixed `k`) is
    not corrupted by also varying total `N`.
18. As a researcher, I want a capacity ladder that runs the **MLP-head rung first**, so
    that any later MIL gain is attributable to aggregation rather than to a bigger head.
19. As a researcher, I want the tile encoder **frozen** in every rung (PRISM and MIL), so
    that the ladder isolates aggregation capacity and never becomes a trainable-encoder
    experiment.
20. As a researcher, I want the MIL rung explicitly **blocked** on tile-feature extraction
    (only slide-level PRISM exists today), so that nobody assumes data we do not have.
21. As a researcher, I want a contingent shared-subspace probe that runs **only if** the
    pooled base shows negative transfer, so that I do not spend effort on a method with no
    problem to solve.
22. As a researcher, I want the probe's win condition fixed in advance (recover the
    pooled-base curve to ≈ the best single-source base), so that "it worked" is falsifiable.
23. As a researcher, I want the probe to learn its subspace from **source organs only** and
    zero-shot the target, so that invariance is never fit using the target (which would
    cheat).
24. As a researcher, I want UCEC-rescue kept as a *bonus* readout of the probe rather than
    its justification, so that the contingent experiment is not tied to the parked
    mechanism question.
25. As a researcher, I want the value substrate to be **parameter-light, fixed-HP, with no
    per-run tune split**, so that scarce labels are not burned on tuning and the comparison
    to the gate stays apples-to-apples.
26. As a researcher, I want the NN-trainer substrate (with train/tune split and
    checkpoint-on-tune-loss) used **only** in the label-rich capacity ladder, so that
    selection variance never contaminates the value measurement.
27. As a researcher, I want `class_weight="balanced"` held fixed across every arm and `k`,
    so that imbalance handling is never itself a confound along the sweep.
28. As a researcher, I want **no SMOTE/resampling** in the value substrate, so that I do
    not add a stochastic, leakage-prone knob exactly where the primary claim is weakest.
29. As a researcher, I want raw pooled OOF AUC as the primary metric with rank-normalized
    pooling reported as a **pre-specified** sensitivity analysis, so that I never switch the
    headline metric post-hoc after seeing a label-informed divergence.
30. As a researcher, I want all results emitted in one tidy schema (`experiment, source,
    target, base, arm, k, draw_seed, auc, …`), so that every figure and significance test
    reads from the same table.
31. As an implementing agent, I want per-component test seams (draw sampler, OOF pooling,
    lift statistic, permutation null), so that each statistical commitment is enforced by a
    test rather than living only in prose.
32. As an implementing agent, I want the draws and permutations seeded deterministically,
    so that the whole sweep is reproducible regardless of parallel scheduling (as the gate
    already is).
33. As a maintainer, I want the existing gate modules (data loading, the fixed-HP probe,
    CV, metrics) reused rather than reimplemented, so that phase 2 shares the gate's
    confound discipline.
34. As a collaborator reading the repo, I want the PRD's commitments traceable to numbered
    decisions in [experimental-design.md](experimental-design.md), so that I can see *why*
    each choice was made.

## Implementation Decisions

**Modules to reuse (existing seams).** Build on the gate library: cohort loading, the
fixed-HP logistic probe (`StandardScaler` + `LogisticRegression(C=1.0,
class_weight="balanced")`), the TSS-grouped CV splitter, and the bootstrap-CI /
permutation-null utilities. Phase 2 adds an E1 scaffold module, a draw/subsampling module,
an injection module (pooling + optional shrinkage), an NN-trainer module (for the ladder),
and per-experiment runner entry points alongside the existing gate runner.

**E1 scaffold (shared site-grouped OOF).** Partition the **target** organ by TSS with
`GroupKFold`. For each held-out fold, draw the `k` target training cases from the *other*
folds' pool and predict the held-out fold; subsample `k` **within** each fold's training
partition — never by globally deleting cases. Pool OOF predictions across folds → **one AUC
per (experiment, source, target, base, arm, k, draw_seed)**. Invariant: the pooled test set
is the full organ at every `k` for every arm.

**`k` semantics (decision 23).** `k` is an **absolute count of added target positives**,
drawn prevalence-matched; grid `k ∈ {0, 3, 5, 10, 25, 50, all}`. `all` = every available
non-test-fold positive (≈80% of the organ; this *is* `cold-start@k=all`, the ceiling).
Negatives per draw = `round(k·(1−p)/p)` at organ prevalence `p` (STAD 0.17, COAD 0.19,
UCEC 0.32). `k=0` = zero-shot.

**Arms and anchors.** `cold-start` = target-only; `warm-start` = source (or pooled source)
∪ `k` target. Anchors fall out of the same scaffold: zero-shot = `warm-start@k=0`;
leave-site-out ceiling = `cold-start@k=all`.

**Two warm-start bases (both run).** Pooled multi-organ base (all-non-target → target) and
single-source base (one foreign organ → target). Headline figure anchored on COAD→STAD;
all matrix cells computed.

**Injection.** Pooling (refit on `source ∪ k target`) is primary. Shrinkage-to-source prior
`λ‖w − w_source‖²` is an optional secondary. Freeze-vs-fine-tune is a no-op for a convex
closed-form fit and is deferred to the MIL regime.

**Significance (decisions 20, 23).** Unit of inference is the paired lift
`Δ(k) = AUC_warm(k) − AUC_cold(k)` on the shared OOF test set, same drawn `k` both arms.

- **Primary statistic:** mean `Δ` over `D` fixed draw seeds; the null recomputes that same
  mean over the same seeds (draws matched, not independent).
- **Confirmatory test:** superiority of `Δ` over zero, **one** pre-registered cell —
  COAD→STAD, pooled base, **`k=10` positives** (principled by a ≈50-case annotation budget;
  not lift-maximizing).
- **Formal null:** within-source label permutation, draws-matched, ~100–1000×; for the
  pooled base, shuffle **within each source organ** (not globally across the concatenated
  source) to preserve per-organ prevalence and composition.
- **Effect-size CI:** paired bootstrap / DeLong on the fixed predictions — a descriptor,
  not the inferential null.
- **Substitution rate `k*`:** reported as an estimate (CI on where warm-start meets the
  ceiling), not a non-inferiority test.
- **Multiplicity:** one confirmatory test; the rest of the matrix shown with CIs, no global
  correction (decision 10's confirmatory/exploratory split).

**Budget-matched swap.** Hold total `N` fixed; trade source for target. Reported as a
distinct question from E1's add-on, on the same scaffold.

**Capacity ladder (decisions 11, 12, 24).** Tile encoder frozen throughout. Order: `linear
head → MLP head` (head capacity; MLP rung first, runs on existing slide vectors) → `frozen
encoder + trainable MIL` (aggregation capacity). Substrate: trained-NN with train/tune
split + checkpoint-on-tune-loss, **label-rich regime only**. **The MIL rung is blocked on
tile-feature extraction** — only 1280-d slide-level PRISM exists today.

**Shared-subspace probe (decisions 13, 21).** Contingent: runs iff the pooled base shows
negative transfer (predicted by `COAD+UCEC→STAD = 0.735 ≈ GI-only`). Linear
domain-adversarial / shared-private projection on frozen PRISM, **source-only fit**, swept
over `k`, evaluated on E1's pooled-base curves against the best single-source base and the
plain pooled-logreg baseline. Win condition: recover the pooled-base curve to ≈ best
single-source base. UCEC-rescue is a bonus readout.

**Cross-cutting (decisions 18, 25, 27, 28, 29).** Substrate decoupled by question; raw
pooled OOF AUC primary (rank-normalized as pre-specified sensitivity, flag if they differ
by >0.01 AUC); `class_weight="balanced"` fixed across all arms and `k`; no
SMOTE/resampling in the value substrate; all output in one tidy results schema.

## Testing Decisions

**What makes a good test here:** assert **external, observable behavior and the
statistical invariants** the design commits to — not internal call structure. Tests must be
**deterministic** (seeded RNG, as the gate already is) and should run on small synthetic or
sub-sampled fixtures so they are fast. Per the chosen *per-component* seam strategy, each
statistical commitment gets its own test point rather than only an end-to-end check.

**Prior art.** There is no test suite yet — this PRD establishes the first one. The model
to follow is the gate's existing determinism: seeded permutations and bootstraps that make
results invariant to parallel scheduling. Reuse the gate's `GroupKFold`-by-TSS splitter and
fixed-HP probe in fixtures so tests exercise the same confound discipline.

**Per-component seams and what each asserts:**

- **Draw / subsampling sampler.** Given a fold training pool, prevalence `p`, and `k`:
  returns exactly `k` positives and `round(k·(1−p)/p)` negatives; never draws from the
  held-out test fold; `k=all` returns every available non-test-fold positive; same seed →
  same indices.
- **OOF scaffold / pooling.** The pooled test set equals the full organ at every `k` and
  for every arm (only training composition changes); `k=0` reproduces the gate zero-shot
  AUC and `k=all` reproduces the cold-start leave-site-out ceiling on a fixture; pooled OOF
  is computed once over concatenated predictions, never as an average of per-fold AUCs.
- **Lift statistic.** `Δ` is paired on the identical drawn `k` cases and identical test
  set across arms; the primary statistic is the mean over `D` fixed seeds and is invariant
  to seed ordering; a shared additive test-time perturbation cancels in `Δ`.
- **Permutation null.** Shuffling labels destroys the feature↔label link while preserving
  prevalence; for the pooled base, shuffling is **within-organ** (per-organ positive counts
  unchanged); the empirical `p` uses the `(1 + #≥obs)/(1 + n)` convention (never `p=0`);
  draws-matched (the same `D` seeds drive observed and null).
- **Injection arms.** Pooling refits on `source ∪ k target`; shrinkage reduces to
  zero-shot as `λ→∞` and to cold-start as `λ→0`.
- **NN trainer (ladder).** Checkpoint selection is on the tune split only; the tile encoder
  is never updated; reproducible under a fixed seed.
- **Metric lock.** Raw pooled OOF is returned as primary; rank-normalized pooling is always
  computed alongside and flagged when it diverges by >0.01 AUC — never silently swapped in.

**Modules tested:** the draw sampler, the OOF scaffold/pooling, the lift statistic, the
permutation null, the injection module, and the NN trainer. The contingent probe gets a
single leakage test (subspace fit touches source only) when/if it is built.

## Out of Scope

- **TP53 transfer atlas** (PRAD/BRCA/BLCA) — blocked on BRCA + BLCA PRISM feature
  extraction; not part of this PRD.
- **Trainable tile encoder** — the ladder freezes the encoder by construction; only the
  aggregator is ever trained.
- **Building the MIL rung** — blocked until tile-level frozen-encoder bags are extracted
  (decision 24). The rung is specified here but not implemented now.
- **PRAD few-shot demonstration (former E2)** — dropped as unmeasurable at n=3 (decision
  22); the measurable version is E1's low-`k` rungs.
- **A cross-pair mechanism regression** — underpowered (n≈1 transferring pair); mechanism
  is parked and kept only as free byproducts.
- **Embedding harmonization (ComBat)** — the gate relies on structural TSS-disjointness;
  no harmonization arm.
- **Per-run hyperparameter tuning / SMOTE in the value substrate** — excluded by
  decisions 18, 25, 28.
- **Re-litigating the gate** — its result is the premise, not under test here.

## Further Notes

- **Gate status:** PASSED (COAD→STAD 0.760, STAD→COAD 0.744, both perm p<0.005; UCEC does
  not transfer). See [results.md](results.md). Honest within-organ ceilings: COAD 0.767,
  UCEC 0.755, STAD 0.858.
- **Sequencing:** E1 first (its `k=0` slice *is* the transfer map, so the map comes free) →
  MLP-head ladder rung (independent of E1 timing) → shared-subspace probe only if the
  pooled base shows negative transfer → budget-matched swap any time after E1. MIL rung
  waits on tile features.
- **Open fork:** MSI-only vs MSI+TP53 scope is `[open]`; recommendation is MSI first as a
  complete story, TP53 as the "second biomarker" follow-up and the route to eventually
  powering the mechanism question.
- **Traceability:** every commitment above maps to a numbered decision (11–25) in
  [experimental-design.md](experimental-design.md).
