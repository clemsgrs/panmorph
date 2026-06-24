# Methods notes

Methodological rationale worth keeping. Backs decisions in
[experimental-design.md](experimental-design.md).

## Leave-site-out CV protocol (the honest within-organ ceiling)

**Goal:** estimate within-organ MSI predictability *without* the within-cohort
tissue-source-site (TSS) shortcut, by never letting a site appear in both train and
test.

### Protocol (agreed)

- **Splitter:** `GroupKFold(n_splits=5)` grouped by TSS. (Optionally also report
  strict leave-one-site-out — one fold per site — as a purest-form robustness check;
  cheap for logistic regression.)
- **Metric:** **pooled out-of-fold (OOF) AUC** — concatenate every case's single
  held-out prediction and compute one AUC over the full cohort. **Never average
  per-fold AUCs.**
- **CI:** patient-level bootstrap 95% CI on the pooled OOF predictions.
- **Guard:** assert each *training* fold has ≥ 10 positives; print per-fold
  test-positive counts for transparency.
- **Do NOT** use `StratifiedGroupKFold` here (see below), and **do NOT** average
  per-fold AUCs (a positive-light fold gives a NaN/garbage per-fold AUC that wrecks
  the mean — pooled OOF is immune to this).

### Why pooled OOF, not averaged per-fold AUC

With pooled OOF, every patient gets exactly one held-out prediction and the AUC is
computed *once* over all 63–155 positives. The per-fold test-positive count then
barely affects metric stability. The only real failure mode is a *training* fold
starved of positives → degenerate classifier; the ≥10-positive guard covers it
(empirically train-positives stay ≥43 per fold in all three MSI cohorts).

### The one assumption pooled OOF makes: cross-fold score comparability

Pooling concatenates predictions from **5 different fold-models** before scoring, so it
quietly assumes those models are **mutually rank-comparable** — a score of 0.7 from
fold-1's model implies roughly the same positive-propensity as 0.7 from fold-2's. This
is *relative* calibration across folds, **not** absolute calibration: AUC is rank-based,
so the scores need not be accurate probabilities, only on a common scale relative to
each other.

Why it matters: pooled AUC is the Mann–Whitney statistic over **all** positive–negative
pairs, and with 5 folds ~80% of those pairs are **cross-fold** (a positive scored by
model *i* vs a negative scored by model *j*). If one fold-model systematically runs hot
(e.g. a shifted intercept from different training-fold prevalence), that bias flows into
the majority of comparisons. Averaged-per-fold AUC is *immune* to this (each AUC uses
one model's ranking only) — but high-variance / undefined with few positives, which is
the bigger threat here and why we still prefer pooled (above).

Why the bias is small in our setup: **same model class, fixed HPs across folds, per-fold
`StandardScaler` (fit on train), and `class_weight="balanced"`** all push the 5 models
onto nearly the same score scale, so cross-fold offset is minor.

**Cheap safeguard (do this):** report **pooled OOF vs averaged-per-fold AUC side by
side**. If they agree, cross-fold calibration drift is a non-issue.

**Locked to avoid a post-hoc metric switch (decision 25).** Raw pooled OOF is the
**primary metric unconditionally** — it is *not* swapped out for a rank-normalized version
on observing the two diverge. Switching the headline metric after seeing a *label-informed*
divergence (AUC reads labels) would be a researcher-degrees-of-freedom problem even though
the rank/quantile normalization itself is label-free. Instead: **always** compute and
report **rank/quantile-normalized-within-fold pooling as a pre-specified sensitivity
analysis** alongside the raw number (it removes between-fold scale offsets while preserving
each model's ranking; label-free, so no leakage). **Flag** if raw and rank-normalized pooled
AUC differ by **> 0.01**; the raw number still stands as primary, the flag just tells the
reader cross-fold calibration drift is non-negligible for that cell.

### Why label-blind GroupKFold balances positives anyway (verified)

Observed test-positives per fold (`GroupKFold(5)`):

- COAD: `[18,18,14,10,14]` (74 total) — balanced
- UCEC: `[32,30,29,34,30]` (155 total) — balanced
- STAD: `[20,12,6,16,9]` (63 total) — fine

`GroupKFold` only equalizes the **number of cases** per fold; it never looks at
labels. Yet positives come out balanced because **each fold is ~20% of the cohort,
and a ~20% chunk of cases carries ~20% of the positives in expectation** — ordinary
proportional sampling. This holds *only because positives are dispersed across many
sites* with no single site monopolizing them (see [data.md](data.md): largest site
holds ≤32% of positives, per-site prevalence in a 13–30% band near the cohort mean).

**Counterfactual that would break it:** one giant site that was, say, 90% MSI+ and
held most positives. Whichever fold inherited that intact site would swallow nearly
all positives, and label-blind GroupKFold could not spread them. *That* regime would
require label-aware grouping. Our data is not in that regime.

### Why `StratifiedGroupKFold` does *worse* here (counterintuitive)

The "obvious" label-aware fix actually creates the degenerate folds it's supposed to
prevent. Observed test-positives (`StratifiedGroupKFold(5, shuffle=True, rs=0)`):

- COAD: `[5,6,8,12,43]` — lumpy
- UCEC: `[2,62,31,20,40]` — one fold n=2
- STAD: `[0,3,8,25,27]` — **zero-positive test fold**

It optimizes a *different* objective (equal class *ratio* per fold) under the hard
"keep sites intact" constraint, via a **greedy heuristic that is not guaranteed
optimal**. With very uneven site sizes (one site = 31% of COAD) and a given seed, the
greedy placement paints itself into a corner and dumps a big high-prevalence site
into one fold. Plain count-balancing has no such constraint to fight and happens to
align better with positive-balancing here.

## Zero-shot defeats the site shortcut "for free"

The MSI cohorts share **zero** TSS (verified in [data.md](data.md)). So in the
zero-shot transfer setting (train source → test target), any site/center signature
the model learned is **out-of-distribution at test time** — the target organ's sites
were never seen. An above-chance zero-shot AUC therefore cannot be a within-cohort
site shortcut. This is a structural argument and is *stronger* than scrubbing site
from embeddings (ComBat/harmonization), which is why we do not harmonize for the gate.

The one residual, essentially untestable risk: a technical artifact correlated with
MSI status *consistently across disjoint cohorts*. The within-source label-permutation
null is what bounds this — a consistent artifact would tend to survive shuffling and
show up as an inflated null.

## Transfer is not a simple shared-direction effect (negative result, do NOT report cosine)

A tempting "mechanistic corroboration" of the zero-shot matrix is to show that the
organs whose probes transfer also point the **same way** in PRISM space — a cosine
between within-organ decision directions. **We checked; it does not hold, and the
naive version actively contradicts the (real) transfer result.** Kept here so nobody
re-derives it and puts it on a slide.

Three variants, all computed on the full-organ probes (seed 0):

| statistic | COAD↔STAD (transfers, AUC .74–.76) | UCEC↔STAD (no transfer, AUC .52) |
|---|---|---|
| LR-weight cosine (`w/σ`, raw space) | +0.10 | +0.03 |
| class-mean-difference cosine (`μ⁺−μ⁻`) | **−0.22** | **+0.37** |
| `cos(w_source, μ⁺−μ⁻ target)`, asymmetric | 0.08–0.11 | 0.00–0.08 |

- **LR-weight cosine** — every pair is ≈orthogonal (0.03–0.10). In 1280-d PRISM space
  with n≈400, `w ≈ Σ⁻¹(μ⁺−μ⁻)` and the ill-conditioned covariance inverse amplifies
  low-variance noise dimensions; `StandardScaler` normalizes per-dim variance but not
  the full covariance, so the direction is noise-dominated and the cosine washes out.
- **Class-mean-difference cosine** — *inverted*: the transferring GI↔GI pair is
  anti-aligned (−0.22) while the non-transferring UCEC↔STAD pair is the most aligned
  (+0.37). Whatever low-noise "MSI shift axis" exists does not predict transfer.
- **Asymmetric `cos(w_source, target shift)`** — the *correct* object (what governs a
  source→target cell is whether the source weight ranks the target's classes). It
  orders the cells roughly right (the two passing cells are the top two) but every
  value is tiny (0.00–0.11) and a failing cell (STAD→UCEC, 0.077) essentially ties a
  passing one (COAD→STAD, 0.082). Not slide-worthy.

**Takeaway.** The transfer is real — AUC, the permutation null, and TSS-disjointness
are untouched by this — but it is **not reducible to a linear-direction-alignment
story in raw embedding space.** The MSI signal lives in a low-variance subspace
swamped by the ambient 1280 dims, so raw-space cosines are near-zero regardless of
whether a pair transfers; only the rank-based AUC surfaces it. Reporting any cosine
number would hand a reviewer an apparent contradiction. If a mechanistic explanation
is ever wanted, it must be done in a **pre-registered** shared low-dim subspace (e.g.
top-K PCA of pooled features) to avoid subspace-fishing — not with raw-space cosines.

## E1 significance: why the source-label-permutation null, not a prediction-swap test

E1's claim is that the **lift** `Δ(k) = AUC_warm(k) − AUC_cold(k)` is real. Two candidate
nulls look similar but test different things:

- **Source-label permutation (chosen).** Shuffle MSI labels *within the source organ*,
  retrain warm-start on the noise-labeled source, recompute Δ vs the real cold-start.
  Null hypothesis: *"the source organ's label↔morphology relationship carries no
  transferable signal."* It **regenerates the pipeline** with source signal destroyed.
- **Prediction-swap permutation (rejected as the inferential null).** Take the two
  fixed prediction vectors (`p_warm`, `p_cold`) and permute the warm/cold tag per
  patient, recomputing Δ. Null hypothesis: *"the two prediction vectors are
  exchangeable"* — given **already-trained** models, are their outputs distinguishable
  on this cohort? It **conditions on the trained models**.

One is about the data-generating mechanism (does source biology transfer?); the other is
about outputs (do these two score vectors differ?). Three reasons the swap test is the
wrong inferential null here:

1. **It ignores the variance that dominates at low `k`.** The swap test conditions on
   one set of trained models — i.e. one particular draw of the `k` local cases — and at
   `k=3` *which* cases you drew is the biggest source of variability in Δ. Holding that
   fixed yields a falsely tight null and over-declares significance. The source-shuffle
   null (matched draws) bakes draw + training variance into the null.
2. **It silently readmits the sample-size confound.** Warm-start has `|source| + k`
   training rows, cold-start has `k`. If warm beats cold, is it source *biology* or just
   thousands of extra rows (of any labeling) stabilizing the model? The swap test cannot
   tell — it attributes any output difference to "value." The source-shuffle null holds
   source data **volume fixed** and randomizes only **labels**, so any pure-volume effect
   is absorbed into the null; only signal-driven excess clears it. This is the same
   sample-size-vs-genuine-signal distinction the whole gate enforces.
3. **Its exchangeability assumption is false even under "no value."** Warm-start trains
   on far more data, so it is intrinsically lower-variance than cold-start (cold-start is
   wild at `k=3`). The two vectors differ in variance structure even when expected AUC is
   equal ⇒ not exchangeable ⇒ the swap null is miscalibrated. The source-shuffle null
   makes no exchangeability assumption; it preserves the warm/cold variance asymmetry by
   construction.

The swap test (equivalently DeLong / a paired bootstrap on the fixed predictions) is
still useful — but as the cheap, **model-conditional effect-size CI** on Δ for a given
draw, *not* as the significance null. Use both: source-label permutation = the formal,
confound-controlling significance test; paired bootstrap/DeLong = the effect-size
descriptor. In one line — the swap test asks "do these two models score differently?";
the source-shuffle null asks "did the foreign organ's biology add value beyond just
having more data?", and only the latter controls the sample-size confound while
capturing few-shot draw variance.

## Label imbalance: weight the estimator, don't rebalance the draw

MSI prevalence is **moderate** (UCEC 32%, COAD 19%, STAD 17% — see [data.md](data.md)),
not extreme. Two things wear the word "imbalance" and must be kept apart:

1. **Estimator handling** — how the model copes with unequal class counts
   (`class_weight`, weighted loss, resampling, threshold). A *modeling* knob.
2. **Draw composition** — how the `k` target cases are sampled in E1 (decision 22's
   A/B/C: prevalence-matched / positives-only / balanced). A *cost-model* choice.

They are applied independently — and keeping them separate settles both.

**Handle imbalance by weighting, not by rebalancing the draw.** The tempting "draw
balanced (C) to fix imbalance" is the wrong fix: it discards cheap, informative
negatives from a measurement whose entire point is a labeling-cost curve.
`class_weight="balanced"` achieves the same "don't ignore the few positives" effect
without throwing data away or distorting the cost model. So tackling imbalance keeps
**C dead and A** (prevalence-matched) **right** — it does not reopen the A/B/C choice.

**The decisive reason imbalance *reinforces* A: A holds class balance constant along the
sweep.** Training prevalence as `k` grows:

| draw | prevalence at `k=3` | at `k=100%` |
|---|---|---|
| **A — prevalence-matched** | ~17% (STAD natural) | ~17% — **constant** |
| B — positives-only (keep all neg) | ~1% | ~20% |
| C — balanced | 50% | 50% |

Under B the imbalance regime swings ~1% → ~20% *along the x-axis*, so the curve conflates
"more positives" with "less imbalanced" and any imbalance handling behaves differently at
each point. Under A balance is fixed at every `k`, so `class_weight` does the identical
thing at every point and the curve isolates the one intended variable (label quantity).

**Commitments:**

- **Metric stays AUC.** Ranking-based and prevalence-independent; and the OOF test set is
  always the **full organ** (fixed prevalence) across every arm and `k`, so no comparison
  is a balance artifact. Optionally report AUPRC as a secondary, prevalence-sensitive
  view — not the comparison metric.
- **`class_weight="balanced"`, fixed across all arms and `k`.** Already what the gate and
  the leave-site-out ceiling use; carrying it unchanged keeps it from ever being itself a
  confound. On the capacity-ladder NN substrate the analog is weighted BCE or a balanced
  sampler — same philosophy, same "fixed across conditions" rule.
- **No SMOTE / resampling in the value substrate.** It adds a stochastic knob, a leakage
  surface, and variance exactly at low `k` where the primary claim is weakest — against
  the parameter-light / fixed-HP / no-per-run-tuning commitment (see
  [research-questions.md](research-questions.md), substrate section).
- **The real low-`k` problem is positive *scarcity*, not imbalance** — absolute count, not
  ratio. The remedy is already in the design (average over many prevalence-stratified
  draws), which *reports* that variance rather than papering over it with resampling.
