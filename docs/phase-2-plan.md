# Phase-2 plan (synthesis)

Single-document synthesis of the phase-2 design agreed in the 2026-06-23 grilling
sessions. It is a **map of the whole plan**; the authoritative detail lives in
[research-questions.md](research-questions.md) (RQs + method roadmap),
[experimental-design.md](experimental-design.md) (decisions log, rows 1–25), and
[methods-notes.md](methods-notes.md) (CV / significance rationale). Where this doc and
those disagree, those win — but they shouldn't.

Status labels: **[agreed]** = settled in review; **[open]** = recommendation noted, not
binding; **[contingent]** = runs only if a stated trigger fires.

---

## 0. Where phase 2 starts

The **go/no-go gate PASSED**: confound-free, organ-dependent cross-organ MSI transfer on
frozen PRISM. COAD↔STAD transfers (COAD→STAD 0.760, STAD→COAD 0.744, perm p<0.005);
UCEC transfers in no direction (p≈0.10–0.44) despite being individually predictable
(site-out 0.755). Cohorts share zero TSS and the target is never trained on, so neither
sample size nor a site shortcut explains it. Phase 2 builds on that result; it does
**not** re-litigate it. (See [results.md](results.md).)

---

## 1. Research questions

| RQ | Status | Question | Powered today? |
|----|--------|----------|----------------|
| **RQ1 — Value** | **primary [agreed]** | How many local positives is a foreign organ worth? | **Yes** — measured *within* existing pairs |
| **RQ2 — Representation / capacity** | secondary | Does the transfer survive / improve under a stronger representation? | Yes (label-rich regime) |
| **RQ3 — Mechanism** | aspirational, **blocked on more organs** | What governs *whether* transfer happens? | **No** — n≈1 transferring pair |

The deep "what governs transfer" question is a claim about variation *across* organ
pairs; with one transferring pair (COAD↔STAD) and one non-sharer (UCEC) that is n≈1.
Mechanism is therefore parked and kept alive only as *free byproducts* — never asserted
as a regression. This power limit is the stated motivation for adding organs (TP53 atlas
/ more MSI cohorts), not a weakness to hide.

---

## 2. The experiment spine

### E1 — value measurement (load-bearing) **[agreed]**

Few-shot value **curves on label-rich organs**, scarcity *simulated* by subsampling
target training positives so a real organ-specific ceiling exists to check against.

- **Sweep** `k ∈ {0, 3, 5, 10, 25, 50, all}` **added target positives** (absolute counts,
  not percentages — decision 23; `3%` of STAD's 63 positives ≈ 2 ≠ 3). `k=0` = zero-shot
  (the gate); `k=all` = every available **non-test-fold** positive (≈80% of the organ —
  *this* is `cold-start@k=all`, the ceiling). The low rungs (`k=3–5` positives on
  STAD/COAD) **are** the genuinely-scarce few-shot regime — measured against a real
  ceiling, which a 3-positive organ like PRAD could never provide (see §below, E2 dropped).
- **`k` counts positives, prevalence-matched.** Each draw carries the matching negatives
  at the organ's natural rate (`k=3` on STAD ≈ 3 pos + ~15 neg), because an MSI label
  costs one *assay per case* and positives arrive at prevalence — so the unit of cost is
  the case, and total `N` stays defined for the swap.
- **Hold source fixed, vary only target `k`** — the curve's x-axis is *target* labels.
- **Shared site-grouped OOF scaffold** (the comparability guarantee). Partition the
  target organ by site (`GroupKFold` over TSS). For each held-out fold, draw the `k`
  target cases from the **other** folds and predict the held-out fold; **subsample `k`
  *within* each fold's training partition, never by globally deleting cases.** Pool OOF
  predictions → one AUC per (arm, `k`). The **pooled test set is the full organ at every
  `k`, for every arm** — only training composition changes — and it is site-clean.
- **Prevalence-stratified subsampling, averaged over many draws** (at `k=3`, *which*
  positives you keep dominates).
- **Anchors fall out of the same scaffold:** zero-shot = `warm-start@k=0`; leave-site-out
  ceiling = `cold-start@k=all`. Same test set, same site discipline as the curves.
- **Injection = pooling** (refit on `source ∪ k target`) [agreed primary]; shrinkage-to-
  source prior (`λ‖w−w_source‖²`) optional secondary. Freeze-vs-fine-tune is a no-op for
  a convex closed-form fit → **deferred to the MIL regime**.
- **Two warm-start bases, both run:**
  - **Pooled multi-organ base** — "is a strong general model worth warm-starting from?"
    Also the baseline the contingent shared-subspace probe is judged against.
  - **Single-source base** — "*which* foreign organ carries the value?" Per-pair
    decomposition; the low-`k` lift is the *free, powered* remnant of RQ3.
- **Measured over the full pair matrix.** Anchor the headline figure on COAD→STAD
  (most lift to show); compute all cells.

**Significance — the lift test [agreed].** Unit of inference is the paired lift
`Δ(k) = AUC_warm(k) − AUC_cold(k)`, same drawn `k` cases feeding both arms; Δ
self-controls for artifacts (test-time leakage inflating both arms cancels).

- **Confirmatory test = superiority over cold-start**, pre-registered: Δ at **`k=10`
  positives** on **COAD→STAD, pooled base** (decision 23 — one rung, not 7 correlated
  per-`k` tests). `k=10` is **principled by a realistic ≈50-case local annotation budget**
  (cost unit = assayed case; at MSI prevalence ≈17–20% that's ≈10 positives — STAD's 17%
  makes 50 cases ≈8–9 pos, `k=10`≈59 cases the nearest rung), *not* chosen to maximize the
  lift. **Primary statistic = mean Δ over `D` fixed draw seeds**; the null
  recomputes that same mean over the same seeds, so draws are matched, not independent
  evidence.
- **Formal null = within-source label permutation, draws-matched** — shuffle MSI labels
  in the source, retrain warm-start, recompute Δ vs the real cold-start, ~100–1000×.
  Holds source *volume* fixed and randomizes only *labels*, so it **controls the
  sample-size confound** (a significant Δ means biology, not "more rows"). **For the pooled
  base, shuffle within each source organ** (preserves per-organ prevalence + composition),
  not globally across the concatenated source (decision 23). This is *not*
  a prediction-swap permutation — see [methods-notes.md](methods-notes.md) for the three
  reasons that null is wrong here (ignores low-`k` draw variance; readmits the confound;
  false exchangeability).
- **Effect-size CI = paired bootstrap / DeLong** on the fixed predictions — descriptor,
  not the inferential null.
- **Substitution rate `k*` = estimation** (CI on where warm-start meets the ceiling), not
  a non-inferiority test (which would need an arbitrary equivalence margin).
- **Multiplicity = confirmatory/exploratory split (decision 10):** one pre-registered
  confirmatory test; rest of the matrix shown with CIs, no global correction.

### E2 (PRAD demonstration) — dropped **[agreed]**

A separate "warm-start PRAD (~3 MSI positives) from the pooled base → usable classifier"
demonstration was **cut**. With 3 positives there is **no test set to measure on**: the
AUC CI spans ≈[0.3, 1.0] and one label flip dominates, so "we get a good classifier" is
unfalsifiable. Its measurable content is **E1's lowest-`k` rungs** (≈2–3 positives on
STAD/COAD against a real ceiling). PRAD is not retained, even as an anecdote. (Decision 22.)

### Budget-matched swap — secondary dual control **[agreed]**

Hold total `N` fixed, *trade* source for target: "at equal labeling cost, is a local case
worth more than a foreign one?" **Distinct from E1's add-on** (source fixed, *add* target)
and reported as a separate question — not folded into E1, which would corrupt the one
number E1 delivers (foreign-data value at fixed `k`).

### Capacity ladder — primary representation/capacity probe (RQ2) **[agreed]**

Tile **encoder frozen throughout**; the variable is tile→slide aggregation. Two separable
limits, run in this order:

1. **Head capacity** — `linear head → MLP head` on the same frozen PRISM vector.
   **MLP rung runs FIRST** (cheap; settles whether a linear boundary is enough).
2. **Aggregation capacity** — frozen-encoder + **trainable MIL** aggregator (not a
   trainable encoder). Running the MLP rung first makes any MIL gain attributable to
   *aggregation*, not "bigger head." **Blocked on tile-feature extraction** — only
   slide-level PRISM exists today, MIL needs tile bags (decision 24; see [data.md](data.md)).
   The MLP rung runs on existing slide vectors and is unblocked.

Substrate: trained-NN with **train/tune split + checkpoint-on-tune-loss**, in the
**label-rich** regime only. Scope: capacity/robustness control, **not** the method
headline. Prior: head capacity is likely not the bottleneck; MIL buys a modest
aggregation gain at most. Interesting exception: if MIL **cracks UCEC**, the
GI↔endometrium wall was an aggregation artifact, not biology.

### Shared-subspace probe — contingent negative-transfer fix **[contingent]**

**Runs iff E1's pooled-base arm shows negative transfer** (predicted by
`COAD+UCEC → STAD = 0.735 ≈ GI-only`). Win condition: a source-only shared-subspace
projection that recovers the **pooled-base curve to ≈ the best single-source base** —
ignoring the non-sharer instead of absorbing its noise; a direct rebuttal of Lee et al.'s
"multi-tissue training is a wash." Falsifiable UCEC-rescue is a **bonus**, not the
justification (the capacity ladder is the primary UCEC-capacity test). Demoted from
pillar because it is underpowered as a standalone (~2 sharing organs) and was tied to the
parked mechanism RQ. Implementation, if triggered: a linear domain-adversarial / shared-
private projection on frozen PRISM, source-only fit, evaluated on E1's pooled-base curves.

---

## 3. Cross-cutting methodological commitments **[agreed]**

- **Substrate decoupled by question.** Value measurement (E1) = parameter-light /
  fixed-HP / **no per-run tune split** (CV over training folds where selection is needed)
  — a held-out tune set has no clean source under scarce-transfer (target burns labels,
  source selects the wrong organ, test leaks). Capacity ladder = trained-NN + tune-split.
  Build the NN trainer once; it is *not* the value-measurement instrument.
- **Confound discipline carries over from the gate.** Structural TSS-disjointness (no
  harmonization); site-grouped CV; the permutation-null philosophy (now the lift null).
- **Pooled OOF AUC, never averaged per-fold — and locked as primary.** Assumes cross-fold
  score comparability (relative, not absolute calibration); fixed-HP + per-fold
  `StandardScaler` + `class_weight="balanced"` keep the 5 models near-common-scale.
  **Raw pooled OOF is the primary metric unconditionally** (decision 25); rank-normalized-
  within-fold pooling is a **pre-specified sensitivity analysis reported alongside**, not a
  post-hoc switch triggered by observing divergence. Flag if they differ by > 0.01 AUC.
  (See [methods-notes.md](methods-notes.md).)
- **Why "training is cheap" matters:** it licenses the full sweep
  (`folds × k × draws × pairs × bases × nulls`) on the parameter-light substrate — not a
  cheaper CV scheme (OOF *is* the 5-fold CV, just scored by pooling).

---

## 4. Anticipated results (what each experiment should show)

| Experiment | Expected outcome | What it buys |
|---|---|---|
| **E1 — COAD→STAD** | Large low-`k` lift over cold-start, converging toward the ~0.76 ceiling by mid-`k`; lift significant vs source-permutation null | The substitution rate + confirmatory value claim |
| **E1 — UCEC→anything** | Warm-start ≈ cold-start (≈0 lift) at every `k` | Free graded transfer readout (parked-RQ3 byproduct) |
| **E1 — pooled vs single-source base** | Pooled base diluted by non-sharer (`COAD+UCEC→STAD ≈ GI-only`) | Triggers the contingent probe |
| **E1 — low-`k` (≈2–3 pos)** | Warm-start already well above chance where cold-start is flat | The scarce-target claim, *measured* (replaces the dropped PRAD demo) |
| **Capacity ladder** | MLP ≈ linear; MIL a modest aggregation gain at most; *maybe* MIL cracks UCEC | De-risks value as not-a-capacity-artifact, or flips the UCEC story |
| **Shared-subspace probe** (if run) | Recovers pooled-base curve toward best single-source; UCEC stays dead | Negative-transfer fix + biological-not-capacity evidence |

---

## 5. Sequencing & dependencies

1. **E1** (the spine) — its `k=0` slice *is* the transfer map, so the map comes free; its
   low-`k` rungs are the scarce-target claim.
2. **Capacity ladder** — MLP rung, then MIL, in the label-rich regime; independent of E1
   timing.
3. **Shared-subspace probe** — only if E1's pooled base shows negative transfer.
4. **Budget-matched swap** — secondary, any time after E1.

**Blockers / [open] forks:**
- **TP53 atlas / mechanism power** is blocked on **BRCA + BLCA PRISM feature extraction**
  (see [data.md](data.md)). It is the route to actually powering RQ3.
- **MSI-only vs MSI+TP53** scope is **[open]** — recommendation: MSI first as a complete
  story; TP53 as the "second biomarker" follow-up.

---

_Synthesized 2026-06-23 from the phase-2 grilling sessions. Tracks decisions 11–25 in
[experimental-design.md](experimental-design.md)._
