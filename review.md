# Documentation and Experiment Plan Review

## Findings

1. **E1 `k` semantics are inconsistent and need tightening before implementation.**
   `docs/phase-2-plan.md` and `docs/research-questions.md` define `k` as
   `{0,3,5,10,25,50,100}%` of target positives, but examples treat `k=3` as
   3 positives. For STAD, 3% of 63 positives is about 2, not 3; for a held-out fold,
   the training partition only has about 80% of positives, so `k=100% of target
   positives` is impossible unless it means "all available non-test positives."

   Recommendation: define `k` as either absolute positive counts or percentages of
   the fold training partition, specify rounding, and pre-register the exact
   confirmatory rung.

2. **The E1 repeated-draw statistic is under-specified.**
   The plan says the permutation null is "draws-matched," but not whether the primary
   statistic is mean Delta over fixed draw seeds, median Delta, pooled predictions over
   draws, or draw-level Deltas. Treating draws as independent would inflate evidence.

   Recommendation: pre-register something like "mean Delta over N fixed draw seeds;
   each permutation recomputes the same mean over the same draw seeds."

3. **Pooled-source label permutation needs a stricter definition.**
   The current gate helper permutes concatenated source labels globally. That is fine
   for exploratory gate cells, but E1's pooled base becomes central.

   Recommendation: for pooled-source E1, shuffle labels within each source organ, not
   across the pooled source, preserving organ-specific prevalence and source
   composition.

4. **Capacity ladder depends on data not documented as available.**
   The plan calls for MIL over tile features, but `docs/data.md` only verifies one
   1280-d PRISM slide/patient embedding per case.

   Recommendation: mark MIL as blocked unless tile-level frozen encoder features and
   slide bags exist, or add the extraction task explicitly as a dependency.

5. **One stale doc line directly contradicts current status.**
   `docs/experimental-design.md` still says the gate is "Not yet run as a clean
   pipeline," while `docs/results.md` records the completed run.

   Recommendation: update that status line.

6. **The pooled OOF fallback could become post-hoc unless locked down.**
   `docs/methods-notes.md` says to rank-normalize if pooled and fold-averaged AUC
   diverge. Since divergence is label-informed, switching the primary metric afterward
   is risky.

   Recommendation: keep raw pooled OOF as primary and report rank-normalized pooling
   as a sensitivity analysis, or predefine a threshold and rule.

7. **Root README is too thin for reproducibility.**
   `README.md` is only a title and tagline. The actual docs are good, but
   discoverability is weak.

   Recommendation: add links to `docs/README.md`, the gate command, and the phase-2
   plan.

## Overall Assessment

The scientific framing is strong: the plan correctly moves from "does transfer exist?"
to "how much local data is foreign data worth?", keeps mechanism claims underpowered,
and avoids the PRAD n=3 trap. The main thing to fix before coding E1 is statistical
precision: exact `k`, exact draw aggregation, and exact pooled-source null. Those are
the pieces that would otherwise create ambiguity in the central claim.

## Verification

Ran:

```bash
python experiments/run_gate.py --n-perm 2 --n-boot 10 --out /tmp/panmorph-review-gate-tiny
```

It completed end to end and reproduced the documented AUC pattern. The verdict from
that run is not meaningful because two permutations cannot support p-values.

Also started:

```bash
python experiments/run_gate.py --quick --out /tmp/panmorph-review-gate --n-jobs 1
```

That run was stopped after it proved too slow for a lightweight review check; it did
load cohorts, pass the TSS-disjointness check, and reproduce the documented ceiling
values before interruption.
