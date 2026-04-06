# E-Commerce Fraud Detection: Multimodal Transfer Learning Framework

This repository is the thesis implementation of [Capinpin, Kobe Andrew S.](https://github.com/VinnRe), [Cuarto, Mico Raphael F.](https://github.com/oocim), and [Penuliar, Alexander Guille A.](https://github.com/agilap) for the degree Bachelor of Science in Computer Science at Batangas State University - The National Engineering University.

The study is titled Cross-Platform Multimodal Product Fraud Detection Framework Using Ensemble Integration and Transfer Learning for E-Commerce Platforms.

This repository contains the implementation, datasets, notebooks, and study artifacts for the multimodal e-commerce fraud detection framework developed in that thesis. The system combines text, image, and tabular metadata signals, then fuses modality probabilities into a final fraud decision.

## Credits

- Researchers: [Capinpin, Kobe Andrew S.](https://github.com/VinnRe), [Cuarto, Mico Raphael F.](https://github.com/oocim), [Penuliar, Alexander Guille A.](https://github.com/agilap)
- Adviser: Lanie P. Palad
- Institution: College of Informatics and Computing Sciences, Batangas State University - The National Engineering University
- Thesis Date: May 2026


## 1. Project Summary

### Goal
Build and evaluate a fraud detection framework for e-commerce listings using:
- Text modality: hybrid RoBERTa + TF-IDF
- Image modality: ResNet-50
- Metadata modality: classical ML models (Logistic Regression, Random Forest, XGBoost)
- Final decision: weighted late-fusion ensemble

### Main Study Focus
- Compare unimodal and multimodal baselines
- Apply transfer adaptation to local datasets
- Improve fraud-class capture under severe class imbalance
- Report operationally relevant metrics (especially Recall and F1)

## 2. Repository Layout

```text
.
|- src/
|  |- pipeline/      # preprocessing, local prep, pseudo-label and transfer-set builders
|  |- training/      # metadata training scripts (baseline + transfer)
|  |- evaluation/    # local baseline and ensemble evaluation scripts
|  |- inference/     # production-style multimodal inference pipeline
|  |- generation/    # synthetic data generation helpers
|- notebooks/
|  |- training/      # model training notebooks (text/image, baseline + transfer)
|  |- analysis/      # analysis and result notebooks
|- data/
|  |- raw/           # original raw CSV(s)
|  |- processed/     # processed modality datasets
|  |- local/         # optional local-only CSVs
|- docs/
|  |- thesis/        # manuscript files
|  |- slides/        # presentation content
|- requirements.txt
|- .gitignore
```

## 3. System Pipeline

1. Raw CSV ingestion and cleaning
2. Modality-specific dataset creation
3. Optional synthetic augmentation and pseudo-labeling
4. Modality model training
5. Late-fusion ensemble scoring
6. Holdout evaluation and comparative analysis

## 4. Data Files

### Primary Inputs
- `data/raw/Training_Data - Train.csv`

### Processed Datasets
- `data/processed/text_dataset.csv`
- `data/processed/image_dataset.csv`
- `data/processed/metadata_dataset.csv`
- `data/processed/synthetic_text_dataset.csv`
- `data/processed/synthetic_metadata_dataset.csv`

## 5. Environment Setup

### Requirements
- Python 3.10+ recommended
- pip
- Internet access may be needed for model/tokenizer downloads when running some workflows

### Install

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## 6. How To Run

This project supports two workflows:
1. Inference-only (use already trained artifacts)
2. Full training and evaluation (rebuild datasets, train models, and evaluate)

### A) Quick Inference (Final Model Demo)

Run multimodal prediction on a CSV matching the raw schema.

```bash
python src/inference/predict.py \
  --profile local_only \
  --input "data/raw/Training_Data - Train.csv" \
  --output "predictions/fraud_predictions.csv"
```

Supported profiles:
1. global
2. local_only
3. ablation

### B) Artifact Directory Contract (Required by predict.py)

The inference loader expects each profile directory to follow this exact structure.

```text
saved_models/ or transfer_models/<profile>/
|- text/
|  |- hybrid_roberta_tfidf.pth
|  |- tfidf_vectorizer.joblib
|  |- tokenizer/...
|- image/
|  |- best_resnet50.pth
|- metadata/
|  |- xgboost.joblib OR random_forest.joblib OR logistic_regression.joblib
|  |- scaler.joblib
|  |- minmax_stats.json
|- ensemble_weights.json   # optional, but recommended
```

If a modality folder is missing, inference still runs but that modality is skipped.

### C) Full Training Workflow (Global Baseline)

Step 1: Rebuild processed modality datasets.

```bash
python src/pipeline/preprocessing_pipeline.py \
  --input "data/raw/Training_Data - Train.csv" \
  --output "data/processed"
```

Step 2: Train text and image models using notebooks.

Training notebooks:
1. notebooks/training/train_text_roberta_colab.ipynb
2. notebooks/training/train_image_resnet_colab.ipynb

Save outputs to:
1. saved_models/text/hybrid_roberta_tfidf.pth
2. saved_models/text/tfidf_vectorizer.joblib
3. saved_models/text/tokenizer/
4. saved_models/image/best_resnet50.pth

Step 3: Train metadata baseline model.

```bash
python src/training/train_metadata.py
```

This script writes model files to saved_models/ by default. For inference compatibility, place metadata artifacts under saved_models/metadata/.

Required metadata files for global profile:
1. saved_models/metadata/xgboost.joblib (or random_forest.joblib or logistic_regression.joblib)
2. saved_models/metadata/scaler.joblib
3. saved_models/metadata/minmax_stats.json

Example (PowerShell) to place baseline metadata artifacts into the expected folder:

```powershell
New-Item -ItemType Directory -Force -Path "saved_models/metadata" | Out-Null
Move-Item "saved_models/xgboost.joblib" "saved_models/metadata/xgboost.joblib" -Force
Move-Item "saved_models/random_forest.joblib" "saved_models/metadata/random_forest.joblib" -Force
Move-Item "saved_models/logistic_regression.joblib" "saved_models/metadata/logistic_regression.joblib" -Force
Move-Item "saved_models/scaler.joblib" "saved_models/metadata/scaler.joblib" -Force
Move-Item "saved_models/minmax_stats.json" "saved_models/metadata/minmax_stats.json" -Force
```

Step 4: Run baseline local evaluation and ensemble evaluation as needed.

```bash
python src/evaluation/evaluate_local_baseline.py --split test
python src/evaluation/ensemble_evaluate.py --profile global
```

### D) Full Training Workflow (Transfer: local_only)

Step 1: Prepare local modality datasets.

```bash
python src/pipeline/prepare_local_datasets.py \
  --local1 "data/local/local1.csv" \
  --local2 "data/local/local2.csv" \
  --out "processed_data/local"
```

Step 2: Create untouched final holdout split.

```bash
python src/pipeline/create_local_final_holdout.py --test-size 0.2 --seed 42
```

Step 3: Generate pseudo labels (repeat by round if needed).

```bash
python src/pipeline/generate_local_pseudolabels.py \
  --round 1 \
  --fraud-min 0.58 \
  --legit-max 0.24 \
  --agreement 2 \
  --max-legit-ratio 3.0
```

Step 4: Build transfer-training modality datasets.

```bash
python src/pipeline/build_local_transfer_training_sets.py --exclude-holdout
```

Step 5: Train transfer text and image models using notebooks.

Training notebooks:
1. notebooks/training/train_text_roberta_transfer_colab.ipynb
2. notebooks/training/train_image_resnet_transfer_colab.ipynb

Save outputs to:
1. transfer_models/local_only/text/hybrid_roberta_tfidf.pth
2. transfer_models/local_only/text/tfidf_vectorizer.joblib
3. transfer_models/local_only/text/tokenizer/
4. transfer_models/local_only/image/best_resnet50.pth

Step 6: Train transfer metadata model.

```bash
python src/training/train_metadata_transfer.py \
  --profile local_only \
  --data "processed_data/local/transfer_train/local_transfer_train_metadata_dataset.csv" \
  --pred-out-dir "predictions/transfer/local_only" \
  --model-out-dir "transfer_models/local_only/metadata"
```

Step 7: Evaluate ensemble for transfer profile.

```bash
python src/evaluation/ensemble_evaluate.py \
  --profile local_only \
  --pred-dir "predictions/transfer/local_only" \
  --output "predictions/transfer/local_only/ensemble_predictions_transfer.csv"
```

### E) Full Training Workflow (Transfer Ablation)

Build mixed local+former datasets, then train using profile ablation paths.

```bash
python src/pipeline/build_local_global_ablation_sets.py \
  --former-ratio 0.4 \
  --local-ratio 0.6 \
  --seed 42
```

Then train/save artifacts under transfer_models/ablation/ and predictions/transfer/ablation/.

## 7. Expected Outputs

Primary output roots:
1. predictions/
2. predictions/local/
3. predictions/transfer/local_only/
4. predictions/transfer/ablation/
5. saved_models/
6. transfer_models/

Typical artifacts:
1. Per-modality probability CSVs
2. Ensemble prediction CSVs
3. Metrics JSON files
4. Experiment logs
5. Trained model binaries and preprocessing assets

Key files produced by scripts:
1. src/training/train_metadata.py -> predictions/metadata_test_predictions.csv and saved_models/*.joblib
2. src/training/train_metadata_transfer.py -> transfer_models/<profile>/metadata/*.joblib and metadata_test_predictions_transfer.csv
3. src/pipeline/generate_local_pseudolabels.py -> predictions/local/local_unlabeled_pseudo_*.csv and processed_data/local/local_pseudo_labeled_raw.csv
4. src/pipeline/create_local_final_holdout.py -> processed_data/local/final_holdout/*

## 8. Reproducibility Notes

- Several scripts use `seed=42` by default.
- Evaluate using an untouched holdout split for fair comparison.
- In fraud-imbalanced setups, prioritize Recall and F1 for operational analysis.

## 9. Known Path Assumptions

Some scripts were originally authored before folder refactoring and still assume legacy directories (for example `processed_data/`, `saved_models/`, `transfer_models/` as runtime roots).

If a script fails with path errors:
- pass explicit path arguments when supported
- create expected runtime folders before running
- or execute the equivalent notebook workflow in `notebooks/`

## 10. License and Usage

Use this repository for academic and research purposes according to your institution's policy and any dataset usage restrictions.
