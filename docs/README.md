# panmorph — project docs

Cross-organ transfer for genomic biomarker prediction from H&E whole-slide images,
on frozen PRISM slide-level embeddings.

This `docs/` directory is the durable record of **verified facts and agreed design
decisions**. Anything here has been checked against data on disk (with the command
that produced it noted) or explicitly agreed in design review. Scratch notes, slide
decks, and exploratory scripts live outside `docs/` and are not authoritative.

## Index

- [data.md](data.md) — cohort statistics, label counts, feature inventory, site structure.
- [experimental-design.md](experimental-design.md) — the go/no-go gate, its null hypothesis,
  controls, model choices, and the decisions log.
- [results.md](results.md) — **gate results (STRONG PASS)** and the full zero-shot matrix.
- [phase-2-plan.md](phase-2-plan.md) — **single-document synthesis of the full phase-2
  plan**: RQs, the experiment spine (E1, capacity ladder, contingent probe),
  cross-cutting commitments, anticipated results, sequencing. Start here for the whole map.
- [prd-phase-2.md](prd-phase-2.md) — **PRD for the phase-2 build** (problem/solution, user
  stories, implementation + testing decisions, scope). Published as
  [panmorph#1](https://github.com/clemsgrs/panmorph/issues/1) (`ready-for-agent`).
- [research-questions.md](research-questions.md) — phase-2 framing in depth: the **value**
  RQ (E1 measurement, low-`k` rungs as the scarce-target regime), substrate decoupling,
  and the method roadmap (shared-subspace probe; PRISM-vs-MIL capacity ladder).
- [methods-notes.md](methods-notes.md) — methodological rationale worth keeping
  (leave-site-out CV protocol; why GroupKFold balances positives here; the E1 lift null;
  label-imbalance handling).
- [related-work.md](related-work.md) — external-literature summaries and how they
  bear on panmorph (Lee et al. 2025 multi-cancer MSI screening).

## One-paragraph status

The **go/no-go gate has PASSED.** Zero-shot transfer (train on source organ(s), test
on a never-seen target organ; fixed-HP logistic probe on frozen PRISM) shows genuine,
confound-free cross-organ MSI transfer for the confirmatory GI↔GI pair (COAD→STAD
0.760, STAD→COAD 0.744, both perm p<0.005), while the gynecological swing organ UCEC
does not transfer (all p≈0.10–0.44). Cohorts share zero tissue-source-sites and the
target is never trained on, so neither sample size nor a site shortcut explains it.
Headline: **cross-organ MSI transfer is real but organ-dependent.** Phase 2 is framed
around the **value** question — *how many local positives is a foreign organ worth?* —
measured by E1 (subsampling on label-rich organs against a real ceiling; its lowest-`k`
rungs are the genuinely-scarce few-shot regime). The mechanism question is demoted to
aspirational, blocked on adding organs. See
[research-questions.md](research-questions.md) and [results.md](results.md).

_Last verified: 2026-06-23._
