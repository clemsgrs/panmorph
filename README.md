# panmorph

**PanMorph: Mapping Cross-Organ Transfer in Histology-Based Genomic Biomarker Prediction.**

Cross-organ transfer of genomic-biomarker prediction (MSI, later TP53) from H&E
whole-slide images, on **frozen PRISM slide-level embeddings** (1280-d, one vector per
case). The go/no-go gate has **PASSED**: confound-free, organ-dependent cross-organ MSI
transfer (COAD↔STAD transfers; UCEC does not). Phase 2 measures the **value** question —
*how many local positives is a foreign organ worth?*

## Documentation

The `docs/` directory is the authoritative record of verified facts and agreed design
decisions. **Start at [docs/README.md](docs/README.md)** for the full index. Key entries:

- [docs/phase-2-plan.md](docs/phase-2-plan.md) — single-document synthesis of the phase-2
  plan (start here for the whole map).
- [docs/experimental-design.md](docs/experimental-design.md) — the gate, its controls, and
  the decisions log (rows 1–25).
- [docs/results.md](docs/results.md) — gate results (STRONG PASS) and the zero-shot matrix.
- [docs/data.md](docs/data.md) — cohort statistics, label counts, feature inventory.
- [docs/methods-notes.md](docs/methods-notes.md) — CV / significance / imbalance rationale.

## Layout

- `src/panmorph/` — library (data loading, CV, the fixed-HP probe, metrics, site probe).
- `experiments/` — runnable entry points (see below).
- `results/` — committed gate output (`full.log`, `gate_results.csv`).

## Running

```bash
# Full gate: zero-shot matrix + within-source permutation null + leave-site-out ceiling
python experiments/run_gate.py                 # 1000 perms, 2000 bootstraps
python experiments/run_gate.py --quick         # smoke test: 100 perms, 500 bootstraps

# Site-decodability diagnostic (not part of the gate decision)
python experiments/run_site_probe.py
```
