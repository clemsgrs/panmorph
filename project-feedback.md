# Cross-Organ Transfer for Genomic Biomarker Prediction from H&E Whole-Slide Images

## Project framing

The project should not be framed primarily as:

> Multi-organ training improves biomarker prediction.

The preliminary MSI results already suggest that this is not reliably true. Depending on the organ combination, multi-organ training may improve, degrade, or leave performance unchanged.

A stronger and more defensible framing is:

> Can we characterize when cross-organ histology training leads to positive transfer, negative transfer, or no transfer for genomic biomarker prediction, and can training objectives reduce negative transfer while preserving useful organ-specific signal?

This turns inconsistent MSI results into the main scientific object of the project.

The core question becomes less about whether MSI or TP53 can be predicted from histology in general, and more about whether a model trained across multiple cancer types learns features that are transferable across organs, organ-specific, or confounded by cohort and acquisition effects.

---

## Current data

Available curated TCGA cohorts:

### MSI

- TCGA-PRAD
- TCGA-UCEC
- TCGA-COAD
- TCGA-STAD

### TP53

- TCGA-PRAD
- TCGA-BRCA
- TCGA-BLCA

---

## Original project idea

The original project investigated whether training deep learning models across multiple cancer types could improve the prediction of genomic biomarkers in prostate cancer.

The initial example was TP53 mutation prediction from H&E-stained whole-slide images. Models were trained on TCGA-PRAD, with and without enrichment from other cancer types such as TCGA-BLCA and TCGA-BRCA. Evaluation sets remained fixed to prostate cancer to assess whether cross-cancer training sharpened biomarker detection in the prostate target domain.

---

## Stage 1: Empirical evidence

### Setup

Preprocessing:

- Combine all slides belonging to the same patient into a patient-level representation.
- Split prostate cancer patients into three cross-validation folds:
  - train
  - tune
  - test
- Augment the training set of each fold with additional breast and bladder cancer patients.

Encoding:

- Extract slide-level features using PRISM.

Training:

- Fit an MLP to map features to TP53 mutation status:
  - `0 = wild type`
  - `1 = mutant`

Evaluation:

- Report AUC on tune and test sets.
- Tune and test sets include only prostate patients.
- Tune and test sets are identical across prostate-only and enriched training setups.

### Initial interpretation

Training on the enriched multi-organ dataset improved prostate performance compared with training on prostate-only data. This suggests that cancers may share predictive morphological signals across organs.

However, MSI experiments showed that this improvement does not necessarily generalize across all biomarkers or organ combinations. This is important and should be central to the project.

---

## Recommended revised hypothesis

Cross-organ histology transfer for genomic biomarker prediction is:

- biomarker-dependent,
- organ-dependent,
- source-target dependent,
- sensitive to class prevalence,
- sensitive to acquisition and site effects,
- and affected by whether the model is forced to learn organ-invariant or organ-aware representations.

Therefore, the goal should be to map and explain transfer regimes rather than to prove that multi-organ training is always beneficial.

---

## What is strong in the project

### 1. PRISM features are a good starting point

Using PRISM slide-level features is a strong starting point because the project is likely limited by sample size and cohort availability. A foundation-model-style slide representation allows you to focus on transfer behavior and training objectives rather than low-level tile encoders.

Reference:

- Shaikovski et al., **PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology**, 2024.  
  <https://arxiv.org/abs/2405.10254>

---

### 2. Using both MSI and TP53 is scientifically useful

MSI and TP53 are different kinds of transfer problems.

MSI has a more plausible cross-organ morphology. MSI/dMMR tumors can show immune-rich morphology, high mutation burden, lymphocytic infiltration, mucinous or medullary patterns in gastrointestinal cancers, and related tumor-microenvironment changes.

TP53 is more context-dependent. A TP53 classifier may learn:

- grade,
- aggressive morphology,
- proliferation,
- genomic instability,
- basal-like phenotype in breast cancer,
- high-grade urothelial morphology in bladder cancer,
- or other organ-specific correlates.

This contrast makes MSI and TP53 a good pair of biomarkers for studying transfer.

---

### 3. Fixed target-organ evaluation is essential

The original prostate setup is methodologically clean because the evaluation set remains fixed to PRAD while the training set changes.

For the broader project, this should be generalized so that every organ becomes the target once.

The key question should be:

> Given a fixed target-organ test set, which source organs help, which hurt, and under which training objectives?

---

### 4. Patient-level aggregation is correct

All slides from the same patient should remain grouped together. Predictions should be evaluated at patient level.

However, instead of describing this as a “packed slide,” it may be clearer to describe it as:

> patient-level aggregation over all available diagnostic slides.

Possible aggregation strategies to compare:

- mean pooling over slide embeddings,
- attention pooling,
- max-logit pooling,
- learned slide-level aggregation.

---

## What needs caution

### 1. More organs should not be assumed to improve AUC

Negative transfer is expected.

A source organ can add useful examples, but it can also introduce:

- contradictory morphology,
- different class prevalence,
- different tumor subtypes,
- organ-specific histologic correlates,
- scanner/site artifacts,
- differences in tumor purity,
- and differences in sampling.

Therefore, “multi-organ training did not improve performance” is not necessarily a failed result. It may be the most interesting result.

---

### 2. MSI in PRAD should be treated carefully

MSI/dMMR is rare in prostate cancer. This means TCGA-PRAD MSI is likely underpowered for stable AUC estimates and should probably not be used as a main MSI target unless there are enough positive cases.

PRAD-MSI can still be useful as an exploratory rare-target experiment, but the main MSI benchmark should probably focus on:

- TCGA-COAD,
- TCGA-STAD,
- TCGA-UCEC.

Reference:

- Study reporting rarity of dMMR/MSI in primary prostate cancer:  
  <https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1277233/full>

---

### 3. Pure organ invariance may be too strong

The original notes propose:

> We expect `zshared` to outperform `zfull` because `zfull` contains organ-specific features that could confuse the model.

This expectation should be softened.

Organ-specific features are not necessarily nuisance. They may be required to interpret the biomarker correctly in each tissue context. For example, MSI morphology in colorectal cancer may not be identical to MSI morphology in endometrial cancer or gastric cancer.

A better expectation is:

> `zshared` should capture transferable signal, while `zspecific` may capture useful organ-conditioned residual signal. The best model may require both.

---

### 4. Adversarial organ removal can hurt

Gradient reversal against organ prediction may remove true biomarker signal if biomarker status is entangled with:

- organ,
- subtype,
- immune phenotype,
- tumor purity,
- grade,
- morphology,
- or prevalence.

Similarly, orthogonality constraints can encourage mathematical separation but do not guarantee biologically meaningful disentanglement.

Organ invariance should therefore be treated as a tunable inductive bias, not as an assumed ideal.

---

### 5. TCGA site confounding is a major risk

TCGA contains site-specific histology signatures. These can bias genomic prediction models if biomarker status, organ, or subtype correlates with tissue source site.

Controls for site and acquisition effects are essential.

References:

- Howard et al., **The impact of site-specific digital histology signatures on deep learning model accuracy and bias**, *Nature Communications*, 2021.  
  <https://www.nature.com/articles/s41467-021-24698-1>

- Evidence that pathology foundation-model embeddings can encode medical center differences:  
  <https://arxiv.org/html/2501.18055v2>

---

## Recommended experiment 1: Transfer atlas

For each biomarker, make every organ the target once.

### MSI

| Target organ | Candidate source organs |
|---|---|
| COAD | UCEC, STAD, PRAD |
| STAD | COAD, UCEC, PRAD |
| UCEC | COAD, STAD, PRAD |
| PRAD | COAD, STAD, UCEC; exploratory only |

### TP53

| Target organ | Candidate source organs |
|---|---|
| PRAD | BRCA, BLCA |
| BRCA | PRAD, BLCA |
| BLCA | PRAD, BRCA |

For each target organ, train on all non-empty source subsets.

For MSI with four organs, this gives seven possible source combinations per target. For TP53 with three organs, this gives three possible source combinations per target.

### Report

For source-only transfer:

```text
AUC(source subset → target)
```

For transfer gain:

```text
Δ transfer = AUC(source subset → target) - AUC(best single-source → target)
```

For target-enriched training:

```text
Δ enrichment = AUC(target + sources) - AUC(target-only)
```

This creates a transfer matrix showing where cross-organ learning is beneficial, harmful, or neutral.

---

## Recommended experiment 2: Few-shot target adaptation

Pure zero-shot transfer may be too strict. A more clinically relevant question is whether source organs help when only a few labeled target-organ cases are available.

For each held-out target organ, train:

1. target-only with 5%, 10%, 25%, 50%, and 100% target labels;
2. source-only with 0% target labels;
3. source plus 5%, 10%, 25%, and 50% target labels;
4. all-source plus target labels;
5. best-source plus target labels.

This answers:

> Does multi-organ training help most when the target organ has very few labels?

This is likely one of the most interesting and useful parts of the project.

---

## Recommended experiment 3: Model ladder

Do not start with the most complex disentanglement model. First establish the transfer phenomenon with simpler baselines.

| Level | Model | Purpose |
|---|---|---|
| 0 | Logistic regression / linear probe on PRISM | sanity baseline |
| 1 | MLP on PRISM | current baseline |
| 2 | Organ-balanced MLP | prevents large organs from dominating |
| 3 | Shared trunk + organ-specific heads | tests shared versus organ-specific prediction |
| 4 | Shared trunk + shared head + organ-conditioned bias | tests calibration and prevalence shift |
| 5 | Mixture-of-experts | allows source organs to contribute differently |
| 6 | Adversarial / CORAL / MMD / GroupDRO variants | domain-generalization baselines |
| 7 | Factorized shared/specific representation | final interpretability-oriented model |

The goal is to determine whether complex disentanglement adds value beyond simple organ-aware modeling.

---

## Revised disentanglement framework

The original framework proposes:

```text
z_full = [z_shared, z_specific]
```

where:

- `z_shared` captures organ-agnostic information,
- `z_specific` captures organ-specific residuals.

A more flexible version is:

```text
z = [z_shared, z_organ, z_residual]
```

### Components

#### 1. Biomarker prediction from shared representation

Use `z_shared` to predict the biomarker. This encourages learning signal that can transfer across organs.

#### 2. Organ prediction from organ-specific representation

Use `z_organ` to predict organ identity. This explicitly captures organ context instead of forcing the entire model to ignore it.

#### 3. Optional adversarial loss on shared representation

Use a gradient reversal layer to reduce organ information in `z_shared`.

This should be tuned carefully. Too much organ removal may remove biomarker signal.

#### 4. Orthogonality or HSIC penalty

Encourage independence between shared and organ-specific branches.

Possible penalty:

```text
z_shared ⟂ z_organ
```

or an HSIC-style dependence penalty.

This should be treated as a regularizer, not proof of biological disentanglement.

#### 5. Organ-balanced supervised contrastive loss

Encourage patients with the same biomarker label from different organs to be closer in representation space, while separating opposite-label patients within and across organs.

This may be more useful than strict organ-adversarial training because it preserves biomarker-relevant cross-organ structure.

---

## Inference comparisons

At test time, compare:

| Representation | Scientific question |
|---|---|
| `z_shared` only | Is there transferable biomarker signal? |
| `z_organ` / `z_specific` only | Is the biomarker mostly organ-dependent? |
| `z_shared + z_specific` | Does organ context help? |
| `z_shared + calibrated organ bias` | Is performance limited by prevalence or calibration shift? |
| full model | What is the best achievable performance? |

A key point:

> It is acceptable if `z_shared + z_specific` outperforms `z_shared` alone.

That would show that robust transfer requires controlled organ specificity rather than pure organ invariance.

---

## Recommended controls

### 1. Site-aware validation

Use TCGA tissue source site where available.

At minimum:

- report biomarker prevalence by site,
- report organ distribution by site,
- check whether label status is site-associated,
- and run site-aware sensitivity analyses.

If sample size allows, create site-grouped or site-held-out splits.

---

### 2. Organ-only and site-only probes

Train simple classifiers from PRISM embeddings to predict:

- organ,
- TCGA tissue source site,
- possibly project/cohort.

If organ and site are nearly perfectly predictable, this is not surprising, but it means that biomarker models may use these factors as shortcuts.

---

### 3. Label-shuffled controls

Shuffle biomarker labels within each organ and rerun the training pipeline.

Expected result:

```text
AUC ≈ 0.5
```

If AUC remains above chance, the model may be exploiting leakage, site effects, or split artifacts.

---

### 4. Clinical and pathology covariate baselines

For TP53 especially, compare the WSI model against simple clinical/pathology covariates such as:

- grade,
- stage,
- subtype,
- tumor purity,
- tumor content,
- available molecular subtype labels.

If WSI adds little beyond these variables, the story changes but remains useful.

---

### 5. Confidence intervals

Report uncertainty using:

- patient-level bootstrapping,
- repeated cross-validation,
- or DeLong confidence intervals.

This is especially important for rare labels such as PRAD-MSI.

Single AUC values will be misleading when the number of positives is small.

---

## Biomarker-specific recommendations

### MSI

The main benchmark should probably focus on:

- TCGA-COAD,
- TCGA-STAD,
- TCGA-UCEC.

TCGA-PRAD MSI should be treated as exploratory unless there are enough positive cases.

The MSI question is strong because MSI is clinically tissue-agnostic in some settings, but its histologic manifestations are not necessarily identical across organs.

A prior multi-cancer MSI study found that tissue-matched models often performed best, while multi-tissue models showed mixed behavior depending on cancer type. This supports the idea that cross-organ MSI transfer is not automatic and should be systematically mapped.

Reference:

- Multi-cancer MSI prediction study:  
  <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0332034>

---

### TP53

The TP53 cohorts are useful:

- TCGA-PRAD,
- TCGA-BRCA,
- TCGA-BLCA.

However, TP53 should be interpreted carefully.

A TP53 classifier may not learn a universal TP53 morphology. It may instead learn organ-specific correlates of TP53 mutation such as:

- aggressive morphology,
- high grade,
- necrosis,
- proliferation,
- basal-like phenotype in breast cancer,
- high-grade urothelial morphology in bladder cancer.

This is not necessarily bad, but the interpretation should avoid overclaiming.

A safer claim is:

> TP53 prediction provides a test case for whether cross-organ transfer is possible for a biomarker whose morphological correlates may be highly organ-specific.

---

## Proposed final project structure

### Aim 1: Build a transfer atlas

Quantify positive and negative cross-organ transfer for MSI and TP53 across all source-target combinations.

Key outputs:

- transfer matrix,
- source-target AUCs,
- Δ transfer,
- Δ enrichment,
- confidence intervals.

---

### Aim 2: Explain transfer behavior

Analyze whether transfer success correlates with:

- biomarker prevalence,
- class imbalance,
- source-target organ similarity,
- embedding similarity,
- immune infiltration,
- tumor purity,
- grade or subtype,
- site effects,
- and batch/acquisition artifacts.

---

### Aim 3: Reduce negative transfer

Compare training strategies:

- target-only baseline,
- pooled multi-organ training,
- organ-balanced sampling,
- shared trunk with organ-specific heads,
- organ-conditioned calibration,
- mixture-of-experts,
- GroupDRO,
- CORAL or MMD alignment,
- adversarial organ-invariant training,
- supervised contrastive learning,
- factorized shared/specific representation learning.

The key question is not simply which model has the highest AUC, but which model reduces negative transfer while preserving positive transfer.

---

### Aim 4: Interpret learned signals

Use interpretability and retrieval analyses to ask whether successful transfer relies on plausible morphology.

Possible approaches:

- high-attention patch retrieval,
- nearest-neighbor retrieval across organs,
- comparison of positive and negative cases,
- expert pathology review of retrieved patches,
- analysis of immune-rich regions for MSI,
- analysis of grade/subtype-associated morphology for TP53.

The goal is to distinguish biologically plausible transfer from shortcuts.

---

## Suggested main conclusion

The strongest version of the project is:

> Cross-organ histology transfer for genomic biomarker prediction is biomarker- and organ-dependent. Some source organs provide useful shared morphological signal, whereas others introduce negative transfer. Robust transfer likely requires organ-aware training that preserves shared biomarker information without erasing useful organ-specific context.

This is stronger than the original claim that multi-organ training generally improves performance.

---

## Immediate next steps

1. Finalize patient-level labels and slide aggregation for each cohort.
2. Count positives and negatives per organ for MSI and TP53.
3. Decide whether PRAD-MSI has enough positives to be included as a main target.
4. Build the full source-target transfer atlas using a simple linear probe and MLP.
5. Add confidence intervals and label-shuffled controls.
6. Only then introduce organ-aware and disentanglement models.
7. Interpret positive-transfer and negative-transfer cases separately.

---

## Minimal experiment table

| Experiment | Purpose | Priority |
|---|---|---|
| Target-only baseline | Establish per-organ predictability | High |
| Source-only transfer | Test zero-shot transfer | High |
| Target + source enrichment | Test whether sources help target prediction | High |
| Few-shot target adaptation | Test low-label benefit | High |
| Organ-balanced training | Control source dominance | High |
| Label-shuffled control | Detect leakage/artifacts | High |
| Site probe | Quantify confounding | High |
| Shared + organ-specific heads | Simple organ-aware model | Medium |
| Mixture-of-experts | Model source-specific usefulness | Medium |
| Adversarial organ-invariance | Test pure invariance | Medium |
| Factorized representation | Final mechanistic model | Medium/Low until baselines are solid |

---

## References

1. Shaikovski et al. **PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology**, 2024.  
   <https://arxiv.org/abs/2405.10254>

2. Howard et al. **The impact of site-specific digital histology signatures on deep learning model accuracy and bias**, *Nature Communications*, 2021.  
   <https://www.nature.com/articles/s41467-021-24698-1>

3. Multi-cancer MSI prediction study, *PLOS One*, 2025.  
   <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0332034>

4. Study reporting rarity of dMMR/MSI in primary prostate cancer, *Frontiers in Oncology*, 2023.  
   <https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2023.1277233/full>

5. Analysis of medical center effects in pathology foundation-model embeddings, 2025.  
   <https://arxiv.org/html/2501.18055v2>
