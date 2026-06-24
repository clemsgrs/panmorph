# Research questions & phase-2 method roadmap

The [gate has PASSED](results.md). This file is the **working framing for phase 2**:
the research questions the gate result licenses, and the method ideas for tackling
them. Two status labels are used:

- **[agreed]** — design decision the team has settled.
- **[open]** — strategic framing not yet decided; a recommendation is noted but it is
  not binding.

## Primary research question **[agreed]**

> **How many local positives is a foreign organ worth?** — when target labels are
> scarce, how much of a local training set can a model pretrained on *other* organs
> substitute for?

This is the **load-bearing question for phase 2.** The gate settled *existence*
(confound-free GI↔GI MSI transfer) and *organ-dependence*: STAD-trained → colon (0.744)
nearly recovers colon's own honest ceiling (0.767), while UCEC fails to transfer in any
direction (all perm p ≈ 0.10–0.44) despite being individually predictable (site-out
0.755). The value question converts that binary into a **continuous substitution rate**
for scarce local labels — the strongest clinical hook, and (critically) one that is
**fully powered by the cohorts we already have**, because it is measured *within*
existing organ pairs rather than regressed *across* a population of pairs. Prior art
(Lee et al. 2025, see [related-work.md](related-work.md)) could not pose it: their
confounded, ImageNet-tile-classifier setup got chance-level cross-tissue everywhere.

**Why value and not mechanism (the honest reason).** The natural deeper question —
*what governs whether transfer happens* — is a claim about variation *across organ
pairs*. With effectively one transferring pair (COAD↔STAD) and one non-sharer (UCEC),
that is **n≈1 of variation**: any "mechanism" asserted from three organs is a story over
three data points, not a tested relationship. Mechanism is therefore **demoted to
aspirational/exploratory and explicitly blocked on adding organs** (TP53 atlas / more
MSI cohorts). This is not a weakness to hide — it is the stated motivation for cohort
expansion. The only powered remnant we keep now is the **binary** falsifiable UCEC
prediction under the shared-subspace probe (rescued or not — see below), which needs no
cross-pair regression.

## Sub-questions

1. **Value (primary) — "how many local positives is a foreign organ worth?"** Full
   measurement design below (E1), including its scarce-target low-`k` regime.
2. **Mechanism (aspirational, blocked on more organs) — why GI↔GI and not
   endometrium?** Underpowered today (n≈1 transferring pair); do **not** assert a
   mechanism regression. Kept alive only as (a) the binary UCEC rescue prediction under
   the shared-subspace probe, and (b) a *free* graded readout from the E1 value curves
   (low-`k` lift per pair — see E1). A real mechanism answer waits for more organs.
3. **Representation / capacity — does the transfer map survive a stronger
   representation?** Two *separable* sub-axes (head capacity vs aggregation capacity);
   see the capacity ladder below.

## The value experiment — E1 **[agreed]**

The value RQ is carried by **one measurement engine, E1**. (An earlier "E2 — PRAD
demonstration" was **dropped**: with 3 PRAD MSI positives there is no test set on which
classifier quality can be measured — any AUC sits on ≤3 held-out positives with a CI
spanning ≈[0.3, 1.0], so the "we get a good classifier" claim is unfalsifiable. The
*measurable* version of that claim — "from a strong base, ~3 local positives already
beat from-scratch" — lives **inside E1** at its lowest `k` rungs, on STAD/COAD where a
real ceiling and a full test set exist. See decision 22.)

### E1 — measurement (load-bearing)

Few-shot value **curves on label-rich organs**, where scarcity is *simulated* by
subsampling the target's training positives so a **real organ-specific ceiling exists
to check against**. This is the rigorous, falsifiable engine that *measures* the
substitution rate — and its low-`k` rungs (`k=3–5%` ≈ **2–3 positives** on STAD/COAD)
are exactly the genuinely-scarce regime, now measured against a checkable ceiling
instead of demonstrated on an unmeasurable one.

- **Protocol.** Train on source organ(s) + `k` added target training cases; sweep `k`
  over **absolute added-positive counts** `{0, 3, 5, 10, 25, 50, all}` (decision 23 — *not*
  percentages; the `%`-vs-count framing was ambiguous, e.g. `3%` of STAD's 63 positives ≈ 2,
  not 3). `k=0` = zero-shot (the gate); `k=all` = every **available non-test-fold** target
  positive (≈80% of the organ — *this* is `cold-start@k=all`, the ceiling), so the sweep top
  is "all the data the scaffold actually exposes," never an impossible 100%-of-organ.
- **What `k` counts (prevalence-matched).** `k` indexes target **positives**, but each
  draw is **prevalence-stratified**, so it carries the matching negatives at the organ's
  natural rate (e.g. `k=3` on STAD ≈ 3 pos **+ ~15 neg**, not 3 pos in isolation and not
  a balanced 3+3; negatives per draw = `round(k·(1−p)/p)` at organ prevalence `p`).
  Rationale: an MSI label is only learned by *assaying* a case, so the
  real unit of labeling cost is the **case** and positives necessarily arrive at
  prevalence — the curve is therefore "what it costs to reach `k` positives," and total
  `N` stays well-defined for the budget-matched swap.
- **Hold source fixed, vary only target `k`** — so the only thing moving along the curve
  is *target* label count, which is exactly the axis "value" is defined on.
- **Comparability is enforced by a shared out-of-fold (OOF) scaffold** (this supersedes
  the looser "thin only the training positives" idea — it was right in spirit but
  underspecified). Partition the **target** organ by site (`GroupKFold` over TSS, as the
  gate ceiling already does). For each held-out fold, draw the `k` target training cases
  from the **other** folds' pool and predict the held-out fold; **subsample `k` *within*
  each fold's training partition, never by globally deleting cases from the organ.** Pool
  the OOF predictions across folds → one AUC per (arm, `k`). Result: the **pooled test
  set is the full organ at every `k`, for every arm** — only the *training* composition
  changes — so all AUCs sit on the identical test set *and* are site-clean (the `k`
  training cases come from sites disjoint from the test fold, so warm-start gets no
  within-organ site boost that the site-clean ceiling lacks). Two AUCs are only
  comparable on the exact same test set; this construction is what guarantees it.
- **Prevalence-stratified subsampling, averaged over a *fixed* set of draws** — at `k=3`,
  *which* three positives you keep dominates the result, so the **primary statistic is the
  mean Δ over `D` fixed draw seeds** (decision 23). The draws are **matched, not
  independent evidence**: the permutation null recomputes that *same* mean over the *same*
  `D` seeds, so averaging tightens the estimate without ever inflating the `p`-value.
- **Imbalance is handled by the estimator, not the draw.** `class_weight="balanced"` is
  held **fixed across every arm and `k`** (as in the gate / ceiling), so it is never
  itself a confound; the prevalence-matched draw is the *only* balance lever. This is also
  why the draw is prevalence-matched and not balanced: A keeps class balance **constant
  along the sweep** (B/C let it drift), so the curve isolates label *quantity*. No SMOTE /
  resampling in this substrate. See [methods-notes.md](methods-notes.md).
- **Every anchor falls out of the same scaffold:** zero-shot = `warm-start@k=0` (source
  only); the **leave-site-out ceiling = `cold-start@k=all`** (target-only, all non-test
  sites). Because they share the partition, the ceiling is a same-test-set,
  same-site-discipline anchor — *not* a standalone full-organ number (which would not be
  comparable to `k>0` points).
- **Injection = pooling (A), primary.** Refit on `source ∪ k target` — same estimator as
  the gate, fewest knobs, apples-to-apples. *Optional secondary:* shrinkage-to-source
  prior `λ‖w−w_source‖²` (B), the literal linear "warm-start" knob (λ→∞ = zero-shot,
  λ→0 = cold-start). The genuine **freeze-vs-fine-tune** fork is **not** meaningful here
  — for a convex closed-form fit the warm-start init doesn't survive optimization — and
  is **deferred to the MIL regime**, where SGD + early-stopping make it a real, distinct
  protocol.
- **Measured over the full pair matrix, not one organ.** The value *curve* is a richer
  transfer readout than the single zero-shot number: the **low-`k` lift**
  (warm-start − cold-start at small `k`) is itself a graded, per-pair transfer measure —
  large for COAD→STAD, ≈0 for UCEC→anything (warm-start collapses onto cold-start). This
  is the free, *powered* remnant of the mechanism question. Anchor the headline figure
  on **COAD→STAD** (most lift to show); compute all cells.
- **Two warm-start bases, two jobs** (run both):
  - **Pooled multi-organ base** (all-non-target → scarce target) — "is a strong general
    model worth warm-starting from?" The "strong general model" arm; also what the
    contingent shared-subspace probe is evaluated against.
  - **Single-source base** (one foreign organ → scarce target) — "*which* foreign organ
    carries the value?" The per-pair decomposition / mechanism-flavored readout.

#### Significance — the lift test **[agreed]**

The curves are not a result until the **lift** is shown to be real. The unit of
inference is the paired lift `Δ(k) = AUC_warm(k) − AUC_cold(k)`, computed on the shared
OOF test set with the **same drawn `k` cases feeding both arms** (so Δ isolates "given
these `k` local labels, does adding source help"). Δ **self-controls for artifacts**:
any test-time leakage inflating both arms cancels in the difference.

- **Confirmatory test = superiority over cold-start (the lift).** Pre-register **one**
  cell (COAD→STAD, pooled base) and **one** summary statistic — Δ at a single
  pre-registered `k`, *not* 7 correlated per-`k` tests. **The rung is `k=10` positives**
  (decision 23), chosen by a *realistic local annotation budget* of ≈50 assayed cases
  (cost unit = the assayed case): at MSI prevalence ≈17–20% that's ≈10 positives (STAD's
  17% → 50 cases ≈8–9 pos, so `k=10`≈59 cases is the nearest grid rung). Principled
  operating point, **not** the `k` that maximizes the lift; the scarcer low-`k` rungs stay
  on the curve as exploratory.
- **Formal null = within-source label permutation, draws-matched** (the direct analog
  of the gate; `k=0` *is* the gate's zero-shot null). Shuffle MSI labels in the source,
  retrain warm-start on the noise-labeled source, recompute Δ vs the *real* cold-start,
  ~100–1000×; matched draws so draw + training variance live in the null. **For the pooled
  multi-organ base, shuffle labels *within each source organ*, not globally across the
  concatenated source** (decision 23): a global shuffle scrambles the organs' differing
  prevalences (UCEC 32% vs STAD 17%) and the source composition the real run preserves;
  within-organ shuffle destroys only the feature↔label link. Real Δ above
  the null's upper tail at p<0.05 ⇒ the foreign organ's *biology* added value **beyond
  just having more training rows** (the null holds source volume fixed, randomizes only
  labels — it controls the sample-size confound; see [methods-notes.md](methods-notes.md)).
- **Effect-size CI = paired bootstrap / DeLong on the fixed predictions** — the cheap,
  model-conditional descriptor of Δ, *not* the inferential null (it ignores draw
  variance and the volume confound — see [methods-notes.md](methods-notes.md) for why a
  prediction-swap permutation is the wrong null here).
- **Substitution rate `k*` = estimation, not a second hypothesis test.** Report a CI on
  where warm-start meets the ceiling, rather than a formal non-inferiority test (which
  would need an arbitrary, defensible-on-demand equivalence margin). Non-inferiority is
  carried only as a robustness row if a reviewer demands a formal equivalence claim.
- **Multiplicity = decision 10's confirmatory/exploratory split.** One pre-registered
  confirmatory lift test; the rest of the matrix (other pairs, single-source bases, full
  curves, UCEC's ≈0 lift) shown with CIs as exploratory, no global correction.

### Why not a separate PRAD demonstration **[agreed]**

The obvious "closing" experiment — warm-start the pooled base on TCGA-PRAD (~3 MSI
positives) and show a usable classifier from a handful of shots — was **dropped**. With
3 positives there is **nothing to measure on**: leave-one-out scores each fold on a
single held-out positive, the AUC CI spans ≈[0.3, 1.0], and *which* 2 of 3 positives
train flips the result. The claim "we get a good classifier" is therefore unfalsifiable
on PRAD itself. Its measurable content — "from a strong cross-organ base, ~3 local
positives already recover a large fraction of the ceiling while from-scratch is at
chance" — is exactly **E1's lowest-`k` rungs on STAD/COAD**, where a full site-clean test
set and a real ceiling exist. PRAD is not carried even as an anecdote (a captioned n=3
scatter still invites the "n=3" dismissal it cannot answer). See decision 22.

### Two value framings that must not be conflated **[agreed]**

- **Add-on (E1):** source fixed, *add* `k` target. "How much does a little local data
  help *on top of* the foreign model?" — the substitution-rate story.
- **Budget-matched swap (decision 7, secondary):** total `N` fixed, *trade* source for
  target. "At equal labeling cost, is a local case worth more than a foreign one?"

Both are "value," but they answer different questions and must be reported as distinct.

## Strategic forks

- **Map vs. method — [resolved toward value].** The phase-2 spine is the **value story
  (E1)**, de-risked by existing results. The cross-organ *map* (zero-shot matrix) is
  the `k=0` slice of E1, so it comes for free. The shared-subspace method is carried as
  the differentiator the value story *earns*, not the thesis it rests on — now scoped as
  a **contingent** experiment (disentanglement is a longer, riskier bet, and underpowered;
  see the probe section below).
- **MSI-only vs. MSI + TP53 — [open].** *Recommendation:* **MSI first, as a complete
  story.** The TP53 atlas is genuinely blocked — BRCA + BLCA PRISM features are not
  extracted (see [data.md](data.md)) — so it cannot be load-bearing now. TP53 is the
  natural "does this generalize to a second, more organ-specific biomarker?" follow-up,
  and (with more organs) the route to actually *powering* the mechanism question.

## Method idea — shared-subspace probe (the disentanglement axis) **[contingent]**

**Status: demoted from pre-committed pillar to a *contingent* experiment** (amends
decision 13). It is **not** the "first phase-2 method experiment" and not on the critical
path. It runs **iff E1's pooled-base arm shows negative transfer worth fixing** — which
the `COAD+UCEC → STAD = 0.735 ≈ GI-only` cell predicts it will. Why demoted: as a
standalone it is underpowered (~2 sharing organs — DANN/IRM have almost nothing to
generalize over), its old headline (UCEC rescue) is attached to the *demoted* mechanism
RQ, and it is redundant with the capacity ladder on that same binary. The one argument
that survives re-attaches it to the **load-bearing value RQ** (below).

**Concept.** Factor the representation into a **shared** (organ-invariant,
biology-carrying) part and **organ-specific** parts; predict from the shared subspace,
and at inference **project the target embedding onto that subspace** rather than using
the whole space. This operationalizes "shared biology" as a concrete object.

**Trigger & primary win condition (the surviving argument).** E1's **pooled-base** arm
warm-starts from all-non-target organs, so a non-sharer in the pool *dilutes the value
curve* (evidence: `COAD+UCEC → STAD = 0.735 ≈ GI-only` — UCEC added nothing). **Trigger:**
run the probe only if this negative transfer shows up in the pooled-base curves.
**Win condition:** a source-only shared-subspace projection that makes the **pooled-base
curve recover to ≈ the best single-source base** — i.e. it *ignores* the non-sharer
instead of absorbing its noise. Measured **inside E1**, against a baseline already
quantified, and a direct, evidenced rebuttal of Lee et al.'s "multi-tissue training is a
wash." It must still *earn* its complexity against plain pooled logreg, which finds a
shared direction implicitly (combined-source cells in [results.md](results.md)).

**Bonus (not the justification) — the falsifiable UCEC mechanism test.** The subspace
preserves/improves GI↔GI transfer at low dimension `k` but **does NOT rescue UCEC** (no
shared GI↔endometrium MSI biology to project onto) → organ-dependence is *biological*; if
it *is* rescued → capacity, not biology, was the wall. Kept as a free readout, **not** the
reason to run the probe — and note the **capacity ladder (decision 12) is the *primary*
"is UCEC a capacity wall" test**; the probe is not a third redundant attempt at it.

**Guardrails [agreed]:**

- **Leakage discipline.** Learn the shared subspace from **source organs only**, freeze
  it, then zero-shot the unseen target — same protocol as the gate. Fitting invariance
  using the target silently cheats.
- **Keep both shared *and* private components.** Forcing everything through an invariant
  bottleneck can destroy organ-specific MSI signal and *lower* within-organ accuracy
  (esp. UCEC, whose signal may be largely private). Use shared for transfer,
  shared+private for within-organ — don't trade the 0.755 UCEC ceiling away chasing
  transfer that may not exist.
- **Underpowered with ~2 sharing organs** — the reason it's contingent, not a pillar.
  Scope as **concept + minimal implementation + falsification**, not a methods bakeoff.
  This is the strongest argument for adding organs (TP53 atlas / more MSI cohorts) —
  state it, don't hide it.

**Implementation, if triggered.** A **linear** shared-subspace probe on **frozen PRISM** —
domain-adversarial linear projection (gradient reversal on an organ classifier) or an
explicit shared/private linear factorization — learned on source organs only, swept over
`k`, evaluated on E1's pooled-base curves against the **best single-source base** and the
**pooled-logreg baseline**. Drops straight into the existing harness. (Later, the same
constraint ports to the trainable MIL aggregator — "make the aggregator emit an
organ-invariant slide embedding" — the more literal form; the frozen-PRISM linear version
is the honest first test.)

## Capacity ladder — PRISM vs MIL **[agreed]**

**Framing.** The tile **encoder is frozen in both regimes**; the variable is the
**tile→slide aggregation**:

- **PRISM** — aggregation (Perceiver tile→slide) is self-supervised-pretrained and
  **frozen**; only the classifier head is trained on a fixed 1280-d slide vector.
- **MIL** — same frozen tile encoder, but a **task-trained** aggregator (frozen
  encoder + trainable MIL, *not* a trainable encoder). **Data dependency:** MIL needs
  frozen-encoder **tile bags**, which are *not yet confirmed extracted* — only 1280-d
  slide-level PRISM exists today (see [data.md](data.md)). The MIL rung is therefore
  **blocked on tile-feature extraction** (decision 24); the MLP-head rung runs on the
  existing slide vectors and is not blocked.

Two **separable** capacity limits, easily conflated:

1. **Head capacity** — linear vs nonlinear head on the *same* frozen PRISM vector. Is
   a linear boundary in PRISM space enough? Cheap (the `torch LP → MLP` rung).
2. **Aggregation capacity** — frozen PRISM aggregation vs trainable MIL over tile
   features. Did PRISM's *task-agnostic* SSL aggregation underweight
   MSI-discriminative signal a label-trained aggregator would keep?

**Ladder order [agreed]:** `linear head → MLP head` (settles #1, **run FIRST**, nearly
free) → `frozen-encoder + trainable MIL` (settles #2). Running the MLP rung first makes
any MIL gain cleanly attributable to **aggregation**, not "bigger head" — otherwise the
MIL comparison is confounded and inconclusive.

**Scope & expectation.** MIL is a **capacity / robustness control, not the method
headline.** Prior: PRISM is a strong slide-level foundation model and MSI is a
slide-level phenotype, so head capacity is likely *not* the bottleneck and MIL buys a
modest aggregation gain at most. The one genuinely interesting exception: if MIL
**cracks UCEC transfer** where frozen PRISM cannot, the GI↔endometrium wall was an
aggregation artifact, not biology → rewrite the mechanism claim (ties back to
sub-question 2).

## Substrate — which estimator answers which question **[agreed]**

Do **not** use one model substrate everywhere; match it to the question and the data
regime.

- **Value measurement (E1)** runs on a **parameter-light, fixed-HP** estimator with
  **no per-run tune split** (sklearn logreg, or an equivalent fixed-reg trained linear —
  same decision boundary). Where model selection is genuinely needed under scarcity, use
  **cross-validation over the training folds**, never a held-out tune set. Reason: in a
  scarce + transfer setting a held-out tune set has **no clean source** — from the target
  it burns the very labels we're economizing; from the source it selects the checkpoint
  best for the *wrong* organ (the zero-shot point); from the test target it leaks.
  Forcing checkpoint selection into E1 injects variance exactly where the primary claim
  is weakest, and breaks apples-to-apples comparison with the gate.
- **Capacity ladder (linear → MLP → MIL)** runs on the **trained-NN substrate *with*** the
  train/tune split + **checkpoint-on-tune-loss** machinery — but in the **label-rich
  regime**, where a tune split is affordable and its job is "is the linear head the
  bottleneck?", *not* "measure value."

Build the NN trainer now (needed for the MLP rung, MIL, and the fine-tune arm), but it is
**not** the measurement instrument for value. Note the genuine freeze-vs-fine-tune
protocol only becomes real on this substrate: SGD + early-stopping make the warm-start
initialization *survive* (implicit regularization toward it), whereas a convex
closed-form fit is init-independent.

_Added 2026-06-23. Reframed around the value RQ (primary), with substrate decoupling,
2026-06-23. E2/PRAD demonstration dropped (unmeasurable at n=3); value carried solely by
E1, low-`k` rungs as the scarce-target regime, 2026-06-23._
