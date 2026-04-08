# MRI Sequence Classifier

A deep learning classifier that identifies brain MRI acquisition types from raw NIfTI files — no metadata, no headers, just the scan.

Supports: **T1w · T2w · FLAIR · DWI · BOLD**

---

## How it works

Upload a `.nii` or `.nii.gz` file and the model returns the predicted sequence type with a confidence score.

Under the hood:
- Extracts the middle axial slice from the 3D volume
- Handles 4D inputs (BOLD) by taking the first timepoint
- Z-score normalizes → clips ±3σ → converts to uint8 RGB
- Runs through a fine-tuned ResNet-18

---

## Model

| Detail | Value |
|---|---|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Frozen layers | layer1, layer2 |
| Output classes | 5 |
| Loss | Weighted CrossEntropyLoss |
| Optimizer | Adam, lr=1e-4 |
| Split strategy | Subject-level GroupShuffleSplit (70/15/15) |
| Test Macro F1 | 1.0 |

Training data: [ds000221](https://openneuro.org/datasets/ds000221), [ds000114](https://openneuro.org/datasets/ds000114), [ds002330](https://openneuro.org/datasets/ds002330) — 250 volumes, 50 per class.

---

## Quickstart

```bash
git clone https://github.com/ayush/mri-sequence-classifier
cd mri-sequence-classifier
pip install -r requirements.txt
streamlit run app.py
```

Place `best_model.pth` in the `models/` directory before running.

---

## Project structure

```
mri-sequence-classifier/
├── app.py              # Streamlit app
├── preprocess.py       # NIfTI preprocessing pipeline
├── requirements.txt
└── models/
    └── best_model.pth
```

---

## Preprocessing pipeline

```
NIfTI load
    → 4D? take first timepoint
    → extract middle axial slice
    → z-score normalize (std + 1e-8)
    → clip ±3σ
    → scale to uint8
    → convert to RGB
    → resize 224×224
    → ImageNet normalize
```

---

## Uncertainty

If the model's top softmax probability is below **0.70**, the prediction is flagged as uncertain rather than returned as a hard label.

---

## Stack

`PyTorch` `torchvision` `nibabel` `Streamlit` `scikit-learn` `Weights & Biases`

---

## Data

All training data sourced from [OpenNeuro](https://openneuro.org). Labels derived from BIDS filenames (`_T1w`, `_T2w`, `_FLAIR`, `_dwi`, `_bold`).

Held-out evaluation on [ds004169](https://openneuro.org/datasets/ds004169) — unseen subjects, unseen dataset.
