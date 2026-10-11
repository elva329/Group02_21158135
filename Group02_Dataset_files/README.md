# BANKING77 Dataset Files

This folder contains the data used by the intent-classification project:

- `train.csv`: 9,002 examples used for fine-tuning.
- `validation.csv`: 1,001 examples held out from the official training split.
- `test.csv`: all 3,080 examples from the official test split.

The training and validation files together reproduce the official BANKING77
training split of 10,003 examples. The final test file was not used for training
or checkpoint selection.

## Columns

- `text`: English banking-support query.
- `label`: integer class identifier from 0 through 76.
- `intent`: human-readable intent name corresponding to `label`.

The validation subset is generated reproducibly from the official training
split using a stratified 10% split and seed 42, matching `train.py`.

## Source and attribution

- Dataset: [PolyAI/BANKING77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)
- Original data repository: [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
- Original paper: Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)
- The dataset script identifies the dataset license as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Retain attribution and verify the license conditions before redistribution.
