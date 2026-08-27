# Project description

Investigates whether training deep learning models across multiple cancer types can improve the prediction of genomic biomarkers in prostate cancer. Specifically, we aim to predict TP53 mutation status from HE-stained whole slide images. Models are trained on TCGA-PRAD (prostate) with and without enrichment from other cancer types, such as TCGA-BLCA (bladder) and TCGA-BRCA (breast). Evaluation sets remain fixed to prostate cancer to assess whether cross-cancer training training can sharpen biomarker detection.

## Stage 1 - Empirical evidence

Setup:

### preprocessing:

- combine all slides belonging to the same patient into a single packed slide
- split prostate cancer patients into three cross-validation folds (train, tune, test)
- augment the training set of each fold with additional breast and bladder cancer patients

### encoding:

extract slide-level features using PRISM

### training

- fit a MLP to map features to TP53 mutation status (0 = wild-type, 1 = mutant)
- report AUC on the tune and test sets, which include only prostate patients and are identical across the prostate-only and enriched training setups

### results 

prostate-only

0.683 ± 0.111 (tune)
0.719 ± 0.070 (test)

enriched

0.722 ± 0.101 (tune)
0.748 ± 0.030 (test)

Training on the enriched (multi-organ) dataset leads to superior prostate performance compared to training on prostate-only data, suggesting that cancers share predictive morphological signals across organs. This motivates the next step: disentangling organ-agnostic from organ-specific features to better understand how cross-organ representations contribute to TP53 prediction.

## Stage 2 - Disentangle organ-agnostic from organ-specific features

The improvement observed when augmenting prostate training data with other organs suggests that cancers share predictive morphological signals across tissue types. However, this also raises a key question: to what extent are these features organ-agnostic vs. organ-specific? To address this, we propose a methodological framework to explicitly disentangle shared, cross-organ representations from organ-dependent ones.

### Factorized Representation Learning

We model the latent feature space as:

zfull = [zshared, zspecific]

where zshared captures organ-agnostic information and zspecific represents organ-specific residuals.

### Training
- prediction from shared features: predict molecular label (e.g., TP53 status) using zshared
- prediction from specific features: predict molecular label using zspecific​ to capture residual discriminative signal
- feature fusion: combine predictions as logitfused = logitshared + αorgan logitsspecific​
- enforce organ-agnostic representations: predict the organ from zshared​ and reverse gradients to suppress organ information
- orthogonality constraint: encourage zshared ⊥ zspecific to enforce independent subspaces

### Inference

At test time, compare performance on a new organ given by zshared vs. zfull 
→ we expect zshared to outperform zfull as zfull contains organ-specific features that could confuse the model