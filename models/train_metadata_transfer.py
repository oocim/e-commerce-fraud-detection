"""Train metadata fraud models on transfer datasets (pure local or ablation mix).

This script mirrors models/train_metadata.py but keeps outputs separate so the
original global training artifacts remain untouched.

Default input:
  processed_data/local/transfer_train/local_transfer_train_metadata_dataset.csv

Ablation input example:
  processed_data/local/transfer_train_ablation_40_60/local_transfer_train_ablation_metadata_dataset.csv

Outputs (default):
    transfer_models/local_only/metadata/metadata_test_predictions_transfer.csv
    transfer_models/local_only/metadata/*.joblib
    transfer_models/local_only/metadata/minmax_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
SEED = 42
N_FOLDS = 5

SMOTE_ENABLED = True
SMOTE_RATIO = 0.3
SMOTE_K_NEIGHBORS = 5


def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
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
    parser = argparse.ArgumentParser(description="Train transfer metadata models")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["local_only", "ablation"],
        default="local_only",
        help="Output profile under transfer_models/",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_DIR / "processed_data" / "local" / "transfer_train" / "local_transfer_train_metadata_dataset.csv"),
        help="Path to transfer metadata dataset CSV",
    )
    parser.add_argument(
        "--pred-out-dir",
        type=str,
        default="",
        help="Directory for transfer prediction outputs (default: transfer_models/<profile>/metadata)",
    )
    parser.add_argument(
        "--model-out-dir",
        type=str,
        default="",
        help="Directory for transfer metadata model artifacts (default: transfer_models/<profile>/metadata)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)

    default_out_dir = PROJECT_DIR / "transfer_models" / args.profile / "metadata"
    pred_out_dir = Path(args.pred_out_dir) if args.pred_out_dir else default_out_dir
    model_out_dir = Path(args.model_out_dir) if args.model_out_dir else default_out_dir

    pred_out_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {data_path}")

    df = pd.read_csv(data_path)
    if "product_id" not in df.columns or "fraud_label" not in df.columns:
        raise ValueError("Dataset must contain product_id and fraud_label columns")

    df["product_id"] = df["product_id"].astype(str)
    df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
    df = df[df["fraud_label"].isin([0, 1])].copy()
    df["fraud_label"] = df["fraud_label"].astype(int)

    feature_cols = [c for c in df.columns if c not in ["product_id", "fraud_label"]]
    X = df[feature_cols].values
    y = df["fraud_label"].values
    pid = df["product_id"].values

    print(f"Loaded transfer metadata dataset: {data_path}")
    print(f"Rows: {len(df)} | Fraud: {int(y.sum())} | Legit: {int((y == 0).sum())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = list(make_models().keys())
    oof_probas = {name: np.zeros(len(df)) for name in model_names}
    oof_preds = {name: np.zeros(len(df), dtype=int) for name in model_names}
    fold_metrics = {name: [] for name in model_names}

    print(f"\nStarting {N_FOLDS}-fold stratified CV ...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'=' * 60}\n  FOLD {fold_idx}/{N_FOLDS}\n{'=' * 60}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if SMOTE_ENABLED:
            n_fraud_before = int(y_train.sum())
            n_legit = int((y_train == 0).sum())
            smote_target = int(n_legit * SMOTE_RATIO)
            if smote_target > n_fraud_before:
                k = min(SMOTE_K_NEIGHBORS, n_fraud_before - 1)
                smote = SMOTE(
                    sampling_strategy={1: smote_target},
                    k_neighbors=k,
                    random_state=SEED + fold_idx,
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                n_syn = len(y_train) - (n_legit + n_fraud_before)
                print(
                    f"  SMOTE: +{n_syn} fraud samples "
                    f"(fraud {n_fraud_before}->{int(y_train.sum())}, total {n_legit + n_fraud_before}->{len(y_train)})"
                )
            else:
                print(f"  SMOTE: skipped ({n_fraud_before} >= target {smote_target})")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        fold_models = make_models()
        for name, model in fold_models.items():
            if name == "Logistic Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

            oof_preds[name][val_idx] = y_pred
            oof_probas[name][val_idx] = y_proba

            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            fold_metrics[name].append({"fold": fold_idx, "f1": f1, "roc_auc": auc})
            print(f"  {name:20s} F1: {f1:.4f} ROC-AUC: {auc:.4f}")

    print(f"\n{'=' * 60}\n  CV RESULTS\n{'=' * 60}")
    results = {}
    for name in model_names:
        y_pred_all = oof_preds[name]
        y_proba_all = oof_probas[name]

        acc = accuracy_score(y, y_pred_all)
        f1 = f1_score(y, y_pred_all)
        roc = roc_auc_score(y, y_proba_all)
        ap = average_precision_score(y, y_proba_all)
        cv_f1_mean = np.mean([m["f1"] for m in fold_metrics[name]])
        cv_f1_std = np.std([m["f1"] for m in fold_metrics[name]])

        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        print(f"Accuracy          : {acc:.4f}")
        print(f"F1 Score (fraud)  : {f1:.4f}")
        print(f"ROC-AUC           : {roc:.4f}")
        print(f"Avg Precision (PR): {ap:.4f}")
        print(f"CV F1 (mean±std)  : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred_all, target_names=["Not Fraud", "Fraud"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred_all))

        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "avg_precision": ap,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
        }

    summary = pd.DataFrame(results).T
    best = summary["f1"].idxmax()
    best_key = best.lower().replace(" ", "_")
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(summary.to_string(float_format="{:.4f}".format))
    print(f"\n>>> Best model by F1: {best} ({summary.loc[best, 'f1']:.4f})")

    preds_df = pd.DataFrame({"product_id": pid, "fraud_label": y})
    for name in model_names:
        key = name.lower().replace(" ", "_")
        preds_df[f"{key}_fraud_proba"] = oof_probas[name]
        preds_df[f"{key}_pred"] = oof_preds[name]
    preds_df["metadata_fraud_proba"] = preds_df[f"{best_key}_fraud_proba"]
    preds_df["metadata_pred"] = preds_df[f"{best_key}_pred"]

    pred_out = pred_out_dir / "metadata_test_predictions_transfer.csv"
    preds_df.to_csv(pred_out, index=False)
    print(f"Saved transfer metadata predictions: {pred_out}")

    print("\nTraining final transfer metadata models on all data ...")
    X_all, y_all = X, y
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    final_models = make_models()
    for name, model in final_models.items():
        if name == "Logistic Regression":
            model.fit(X_all_scaled, y_all)
        else:
            model.fit(X_all, y_all)
        key = name.lower().replace(" ", "_")
        out_path = model_out_dir / f"{key}.joblib"
        joblib.dump(model, out_path)
        print(f"  Saved: {out_path}")

    joblib.dump(scaler, model_out_dir / "scaler.joblib")

    scale_cols = [
        "listed_price", "original_price", "price_deviation", "price_ratio",
        "seller_rating", "rating_count", "item_rating", "item_rating_count",
        "review1_rating", "review2_rating", "review_rating_diff", "seller_item_rating_gap",
    ]
    minmax_stats = {}
    for col in scale_cols:
        if col in df.columns:
            minmax_stats[col] = {"min": float(df[col].min()), "max": float(df[col].max())}
    minmax_path = model_out_dir / "minmax_stats.json"
    with open(minmax_path, "w", encoding="utf-8") as f:
        json.dump(minmax_stats, f, indent=2)

    print(f"Saved min/max scaling stats: {minmax_path}")
    print(f"\nSaved transfer metadata artifacts to: {model_out_dir}")


if __name__ == "__main__":
    main()
