# Cross-organ MSI transfer

PanMorph studies whether labelled pathology data from one organ can help predict MSI in
another organ when local positive cases are scarce.

## Language

**Source organ**:
The organ whose labelled cohort is added to training.
_Avoid_: Foreign organ

**Target organ**:
The organ the model predicts. Labels from this organ are local labels.
_Avoid_: Destination organ

**Zero-shot transfer**:
Training on a source organ and testing on a target organ without target-organ labels.

**Few-label transfer**:
Comparing source-plus-local training with local-only training as a small number of local
MSI-positive cases is added.
_Avoid_: Experiment codes, phase numbers, value experiment

**Local-positive count**:
The number of target-organ MSI-positive cases used for training; matched local negatives
are added at the target organ's observed prevalence.
_Avoid_: k, rung
