# Related work

External-literature summaries and how they bear on panmorph. Unlike the rest of
`docs/`, entries here are **summaries of others' published work plus our reading of
it** — not facts verified against our own data. Treat the comparisons as design
input, not settled results.

## Lee, Kim, Lee & Chung 2025 — multi-cancer histopathologic MSI screening

PLoS One 20(9):e0332034, <https://doi.org/10.1371/journal.pone.0332034>.
PDF: [`bib/journal.pone.0332034.pdf`](../bib/journal.pone.0332034.pdf).

**What it is.** Slide-level MSI-H vs. MSS prediction from H&E across TCGA **CRC
(COAD+READ), STAD, UCEC**, with the explicit question of whether multi-tissue
training generalizes MSI features across organs. External validation on CPTAC-COAD
and CPTAC-UCEC (no CPTAC-STAD).

**Method (the part we'd do differently).** ImageNet-pretrained tile CNNs
(EfficientNet-b0 / ResNet18 / VGG19 / ConvNeXt / NAT, 60 model variants). A ResNet50
tumor-tile classifier first restricts input to tumor tiles; then a **per-tile MSI
classifier** scores each tile and the **slide score = mean of tumor-tile MSI
probabilities**.

**Headline results (AUC).**
- Corresponding-organ: CRC **0.93**, STAD **0.84**, UCEC **0.79** (UCEC always
  weakest). Comparable to / slightly above prior work (Kather 0.77/0.81/0.75).
- **Cross-organ transfer is poor**: CRC→STAD ~0.57, CRC→UCEC ~0.60, UCEC→{CRC,STAD}
  ~0.57. Best cross pair is **STAD→CRC ~0.72**, and it is asymmetric (CRC→STAD only
  ~0.57) — i.e. weak GI↔GI, essentially no GYN↔GI.
- Multi-tissue training is a wash: equal/slightly worse for CRC and STAD; helps only
  UCEC (NAT 0.79 vs 0.69 single-organ), attributed to UCEC's own model being weak.
  A CRC+STAD-only model ≈ the three-tissue model, suggesting UCEC mostly adds noise.

### Overlap with panmorph

Same organs (CRC/STAD/UCEC), same task (MSI-H vs MSS), same label source (PCR on
TCGA). The **data and label side overlaps heavily.** Both works also find **UCEC
(GYN) is the odd one out** — weak and non-transferring — convergent evidence that MSI
morphology is GI-shared but GYN-distinct (cf. our gate: UCEC p≈0.10–0.44).

### Net differences (why our result diverges, and where we go next)

1. **Representation.** They use ImageNet-pretrained tile CNNs + a per-tile MSI
   classifier whose slide score is the mean of tile probabilities — a dated
   comp-path recipe. We dislike it: we only have **slide-level labels, not clean
   per-tile labels**, so a per-tile classifier is the wrong tool. The modern approach
   (and our plan) is a **strong in-domain pathology foundation-model encoder** (not
   ImageNet) **+ MIL aggregation**. PRISM is our baseline to get feasibility numbers;
   the planned transition is **tile encoder + MIL**. Their own "future work" section
   literally lists "use H&E foundation models instead of ImageNet" and "explore MIL"
   — i.e. our starting point is their to-do list.

2. **Transfer outcome.** Their CRC→STAD is ~0.57 (chance); our frozen-PRISM +
   logistic probe gives **COAD→STAD 0.760 / STAD→COAD 0.744, perm p<0.005** — genuine
   GI↔GI transfer. Plausibly a representation effect (foundation embedding vs
   ImageNet tile features). See [results.md](results.md).

3. **Confound control.** We enforce leave-site-out / zero shared TSS between cohorts
   (see [methods-notes.md](methods-notes.md)); they do **not** control the TCGA
   site signature — they even cite Howard et al. 2021 on it in the discussion but
   don't act on it. A defensible novelty axis for us.

### Idea worth borrowing: restrict to tumor tiles

Their tumor-tile gating could raise MSI signal-to-noise by dropping stroma/normal
tissue. **Worth trying in our MIL setting** — *unless* MSI-related signal extends
beyond tumor regions (e.g. peritumoral immune/TIL response), in which case hard
tumor-only masking would discard signal. Open question; candidate ablation:
tumor-only vs all-tissue tiling under the same encoder + MIL head. (Caveat from their
own paper: their tumor classifier was trained on colon/stomach tumor only — **no
UCEC** — so UCEC tumor calls were unreliable; any tumor-gating we adopt must cover
all target organs.)

_Added 2026-06-23._
