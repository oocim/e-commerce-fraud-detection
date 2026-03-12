"""
Train metadata fraud models (Random Forest, Logistic Regression, XGBoost).

This script:
1) Loads metadata_dataset + synthetic_metadata_dataset (fraud-only augmentation)
2) Runs 5-fold stratified CV to generate out-of-fold predictions for ALL
   real samples (matching text & image prediction counts for ensemble)
3) Trains final models on ALL data for deployment
4) Exports cross-validated predictions + saved models & scaler to disk
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR if (BASE_DIR / "processed_data").exists() else BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "processed_data" / "metadata_dataset.csv"
SYNTH_PATH_CANDIDATES = [
    PROJECT_DIR / "processed_data" / "synthetic_metadata_dataset.csv",
    PROJECT_DIR / "synthetic_metadata_dataset.csv",
    PROJECT_DIR / "synthetic_metadata__dataset.csv",
]
OUTPUT_DIR = PROJECT_DIR / "predictions"
MODEL_DIR = PROJECT_DIR / "saved_models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42
N_FOLDS = 5
SYNTH_CAP = 300  # use all 300 template-generated synthetic fraud rows

# SMOTE config (oversamples fraud in numerical feature space)
# Metadata features are pure tabular numerics — the classic SMOTE use case.
# SMOTE interpolates between real fraud feature vectors to create synthetic
# fraud samples, improving minority class representation.
SMOTE_ENABLED = True
SMOTE_RATIO = 0.3          # target fraud ratio (0.3 = ~30% of majority count)
SMOTE_K_NEIGHBORS = 5      # neighbors for interpolation


def make_models():
    """Create fresh model instances (needed for each fold)."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def main() -> None:
    # ── Load data ────────────────────────────────────────────────
    print("Loading training datasets...")
    metadata_df = pd.read_csv(DATA_PATH)
    print(f"Original dataset: {metadata_df.shape}")
    print(f"Fraud distribution:\n{metadata_df['fraud_label'].value_counts()}\n")

    synth_path = next((path for path in SYNTH_PATH_CANDIDATES if path.exists()), None)
    if synth_path is None:
        raise FileNotFoundError("Synthetic metadata dataset not found.")

    synthetic_df = pd.read_csv(synth_path)
    print(f"Synthetic dataset loaded from: {synth_path}")
    print(f"Synthetic shape: {synthetic_df.shape}")

    # Cap synthetic data to prevent overfitting
    if len(synthetic_df) > SYNTH_CAP:
        synthetic_df = synthetic_df.sample(n=SYNTH_CAP, random_state=SEED)
        print(f"Capped synthetic data to {SYNTH_CAP} rows")

    feature_cols = [c for c in metadata_df.columns if c not in ["product_id", "fraud_label"]]

    # Separate real and synthetic for proper k-fold handling
    # (synthetic always in training, never in validation — same as text/image)
    real_df = metadata_df.copy()
    X_real = real_df[feature_cols].values
    y_real = real_df["fraud_label"].astype(int).values
    pid_real = real_df["product_id"].values

    X_synth = synthetic_df[feature_cols].values
    y_synth = synthetic_df["fraud_label"].astype(int).values

    print(f"\nReal data:      {len(real_df)} rows ({y_real.sum()} fraud)")
    print(f"Synthetic data: {len(synthetic_df)} rows ({y_synth.sum()} fraud)")
    print(f"Synthetic data will be added to EVERY training fold (never in validation)")

    # NOTE: No class_weight/scale_pos_weight — synthetic fraud data
    # already boosts fraud representation. Adding class weighting on top
    # causes severe overconfidence and false positives on legitimate listings.

    # ═══════════════════════════════════════════════════════════════
    #  5-FOLD STRATIFIED CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════
    # Each fold:
    #   1. Split REAL data into train/val (synthetic always in train)
    #   2. Fit scaler on training data only
    #   3. Train all 3 models, collect val predictions
    #   4. Result: every real sample gets an out-of-fold prediction
    # ═══════════════════════════════════════════════════════════════

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = list(make_models().keys())
    # Storage for out-of-fold predictions (one array per model)
    oof_probas = {name: np.zeros(len(real_df)) for name in model_names}
    oof_preds = {name: np.zeros(len(real_df), dtype=int) for name in model_names}
    fold_metrics = {name: [] for name in model_names}

    print(f"\nStarting {N_FOLDS}-fold stratified cross-validation ...\n")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_real, y_real), 1):
        print(f'{"="*60}')
        print(f'  FOLD {fold_idx}/{N_FOLDS}')
        print(f'{"="*60}')

        # Split real data
        X_fold_train_real = X_real[train_indices]
        y_fold_train_real = y_real[train_indices]
        X_fold_val = X_real[val_indices]
        y_fold_val = y_real[val_indices]

        # Add ALL synthetic data to training fold
        X_fold_train = np.vstack([X_fold_train_real, X_synth])
        y_fold_train = np.concatenate([y_fold_train_real, y_synth])

        n_train_fraud = int(y_fold_train.sum())
        n_val_fraud = int(y_fold_val.sum())
        print(f"Train: {len(X_fold_train)} ({n_train_fraud} fraud, "
              f"{100*n_train_fraud/len(X_fold_train):.1f}%)")
        print(f"Val:   {len(X_fold_val)} ({n_val_fraud} fraud, "
              f"{100*n_val_fraud/len(X_fold_val):.1f}%)")

        # ── SMOTE oversampling in numerical feature space ──────────────
        if SMOTE_ENABLED:
            n_fraud_before = n_train_fraud
            n_legit = int((y_fold_train == 0).sum())
            smote_target = int(n_legit * SMOTE_RATIO)

            if smote_target > n_fraud_before:
                k = min(SMOTE_K_NEIGHBORS, n_fraud_before - 1)
                smote = SMOTE(
                    sampling_strategy={1: smote_target},
                    k_neighbors=k,
                    random_state=SEED + fold_idx,
                )
                X_fold_train, y_fold_train = smote.fit_resample(
                    X_fold_train, y_fold_train
                )
                n_synthetic = len(y_fold_train) - (n_legit + n_fraud_before)
                print(f"  SMOTE: +{n_synthetic} fraud samples "
                      f"(fraud {n_fraud_before}\u2192{int(y_fold_train.sum())}, "
                      f"total {n_legit + n_fraud_before}\u2192{len(y_fold_train)})")
            else:
                print(f"  SMOTE: skipped \u2014 fraud already \u2265 target "
                      f"({n_fraud_before} \u2265 {smote_target})")
        # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

        # Fit scaler on this fold's training data only
        fold_scaler = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = fold_scaler.transform(X_fold_val)

        # Train and evaluate each model on this fold
        fold_models = make_models()
        for name, model in fold_models.items():
            if name == "Logistic Regression":
                model.fit(X_fold_train_scaled, y_fold_train)
                y_pred = model.predict(X_fold_val_scaled)
                y_proba = model.predict_proba(X_fold_val_scaled)[:, 1]
            else:
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                y_proba = model.predict_proba(X_fold_val)[:, 1]

            # Store out-of-fold predictions
            oof_probas[name][val_indices] = y_proba
            oof_preds[name][val_indices] = y_pred

            fold_f1 = f1_score(y_fold_val, y_pred)
            fold_roc = roc_auc_score(y_fold_val, y_proba)
            fold_metrics[name].append({
                "fold": fold_idx,
                "f1": fold_f1,
                "roc_auc": fold_roc,
            })
            print(f"  {name:20s}  F1: {fold_f1:.4f}  ROC-AUC: {fold_roc:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  AGGREGATED CROSS-VALIDATION RESULTS
    # ═══════════════════════════════════════════════════════════════

    print(f'\n{"="*60}')
    print(f'  {N_FOLDS}-FOLD CV RESULTS (all {len(real_df)} real samples)')
    print(f'{"="*60}')

    results = {}
    for name in model_names:
        y_pred_all = oof_preds[name]
        y_proba_all = oof_probas[name]

        acc = accuracy_score(y_real, y_pred_all)
        f1 = f1_score(y_real, y_pred_all)
        roc = roc_auc_score(y_real, y_proba_all)
        ap = average_precision_score(y_real, y_proba_all)

        per_fold = fold_metrics[name]
        cv_f1_mean = np.mean([m["f1"] for m in per_fold])
        cv_f1_std = np.std([m["f1"] for m in per_fold])

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"F1 Score (fraud)  : {f1:.4f}")
        print(f"ROC-AUC           : {roc:.4f}")
        print(f"Avg Precision (PR): {ap:.4f}")
        print(f"CV F1 (mean±std)  : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_real, y_pred_all, target_names=["Not Fraud", "Fraud"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_real, y_pred_all))

        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "avg_precision": ap,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
        }

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary.index.name = "Model"
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary["f1"].idxmax()
    print(f"\n>>> Best model by F1 score: {best} ({summary.loc[best, 'f1']:.4f})\n")

    # ═══════════════════════════════════════════════════════════════
    #  EXPORT CROSS-VALIDATED PREDICTIONS (all real samples)
    # ═══════════════════════════════════════════════════════════════

    test_predictions = pd.DataFrame({"product_id": pid_real, "fraud_label": y_real})
    for name in model_names:
        key = name.lower().replace(" ", "_")
        test_predictions[f"{key}_fraud_proba"] = oof_probas[name]
        test_predictions[f"{key}_pred"] = oof_preds[name]

    # Use best model's proba as the representative metadata probability
    best_key = best.lower().replace(" ", "_")
    test_predictions["metadata_fraud_proba"] = test_predictions[f"{best_key}_fraud_proba"]
    test_predictions["metadata_pred"] = test_predictions[f"{best_key}_pred"]

    output_path = OUTPUT_DIR / "metadata_test_predictions.csv"
    test_predictions.to_csv(output_path, index=False)
    print(f"Saved cross-validated predictions: {output_path}")
    print(f"Rows: {len(test_predictions)} (all real samples — matches text & image)")
    print(f"Columns: {list(test_predictions.columns)}")

    # ═══════════════════════════════════════════════════════════════
    #  TRAIN FINAL MODELS ON ALL DATA (for deployment)
    # ═══════════════════════════════════════════════════════════════

    print(f"\nTraining final models on ALL data for deployment ...")
    X_all = np.vstack([X_real, X_synth])
    y_all = np.concatenate([y_real, y_synth])
    print(f"Final training set: {len(X_all)} samples ({y_all.sum()} fraud)")

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    final_models = make_models()
    for name, model in final_models.items():
        if name == "Logistic Regression":
            model.fit(X_all_scaled, y_all)
        else:
            model.fit(X_all, y_all)
        key = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{key}.joblib")
        print(f"  Saved: {key}.joblib")

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Save min/max stats for the _scaled features (needed at inference)
    scale_cols = [
        "listed_price", "original_price", "price_deviation", "price_ratio",
        "seller_rating", "rating_count", "item_rating", "item_rating_count",
        "review1_rating", "review2_rating", "review_rating_diff", "seller_item_rating_gap",
    ]
    minmax_stats = {}
    for col in scale_cols:
        if col in metadata_df.columns:
            minmax_stats[col] = {
                "min": float(metadata_df[col].min()),
                "max": float(metadata_df[col].max()),
            }
    minmax_path = MODEL_DIR / "minmax_stats.json"
    with open(minmax_path, "w") as f:
        json.dump(minmax_stats, f, indent=2)
    print(f"Saved min/max scaling stats: {minmax_path}")

    print(f"\nSaved models to: {MODEL_DIR}")
    for f in sorted(MODEL_DIR.glob("*.joblib")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
