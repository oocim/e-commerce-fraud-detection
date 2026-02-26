"""
Train metadata fraud models (Random Forest, Logistic Regression, XGBoost).

This script:
1) Loads metadata_dataset + synthetic_metadata_dataset (fraud-only augmentation)
2) Trains and evaluates three classifiers with stratified split
3) Exports test-set predictions (product_id, fraud_label, probabilities)
   for use in the multimodal ensemble
4) Saves trained models and scaler to disk
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
SYNTH_CAP = 150  # max synthetic fraud rows (prevents overfitting)


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

    combined_df = pd.concat([metadata_df, synthetic_df], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Combined fraud distribution:\n{combined_df['fraud_label'].value_counts()}")
    print(f"Fraud ratio: {100*combined_df['fraud_label'].mean():.1f}%\n")

    feature_cols = [c for c in metadata_df.columns if c not in ["product_id", "fraud_label"]]

    X = combined_df[feature_cols].copy()
    y = combined_df["fraud_label"].astype(int).copy()
    product_ids = combined_df["product_id"].values

    # ── Train / Test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, product_ids, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"Train fraud ratio: {y_train.mean():.4f}")
    print(f"Test  fraud ratio: {y_test.mean():.4f}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    neg, pos = np.bincount(y_train)
    print(f"Class distribution: not-fraud={neg}, fraud={pos} ({100*pos/(neg+pos):.1f}%)")

    # NOTE: No class_weight/scale_pos_weight — synthetic fraud data (500 rows)
    # already boosted fraud from 4.3% to 20.7%. Adding class weighting on top
    # causes severe overconfidence and false positives on legitimate listings.

    # ── Define models ────────────────────────────────────────────
    # Regularization tuned to prevent overfitting on small dataset (~2,575 rows)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,    # reduced from 300
            max_depth=8,         # reduced from 12 — shallower trees generalize better
            min_samples_split=10,  # increased from 5
            min_samples_leaf=5,    # increased from 2
            max_features='sqrt',   # limit features per split
            random_state=SEED,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,             # added L2 regularization (default C=1.0 is too loose)
            solver="lbfgs",
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,    # reduced from 300
            max_depth=4,         # reduced from 6 — prevents memorization
            learning_rate=0.05,  # reduced from 0.1 — slower learning
            min_child_weight=5,  # requires more samples per leaf
            subsample=0.8,       # row subsampling — stochastic regularization
            colsample_bytree=0.8,  # column subsampling
            reg_alpha=0.1,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = {}
    test_predictions = pd.DataFrame({"product_id": pid_test, "fraud_label": y_test.values})

    # ── Train, evaluate, collect test predictions ────────────────
    print("Training and evaluating models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        print("=" * 60)
        print(f"  {name}")
        print("=" * 60)

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        print(f"\nAccuracy          : {acc:.4f}")
        print(f"F1 Score (fraud)  : {f1:.4f}")
        print(f"ROC-AUC           : {roc:.4f}")
        print(f"Avg Precision (PR): {ap:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred), "\n")
        print(f"5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "avg_precision": ap,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        }

        # Store per-model test predictions
        key = name.lower().replace(" ", "_")
        test_predictions[f"{key}_fraud_proba"] = y_proba
        test_predictions[f"{key}_pred"] = y_pred

        # Save model
        joblib.dump(model, MODEL_DIR / f"{key}.joblib")

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Save min/max stats for the _scaled features (needed at inference)
    # The metadata_dataset.csv has pre-computed _scaled columns using the
    # full dataset's min/max. We need those same stats for single-row inference.
    import json
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

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary.index.name = "Model"
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary["f1"].idxmax()
    print(f"\n>>> Best model by F1 score: {best} ({summary.loc[best, 'f1']:.4f})\n")

    # Use best model's proba as the representative metadata probability
    best_key = best.lower().replace(" ", "_")
    test_predictions["metadata_fraud_proba"] = test_predictions[f"{best_key}_fraud_proba"]
    test_predictions["metadata_pred"] = test_predictions[f"{best_key}_pred"]

    # ── Export test predictions for ensemble ──────────────────────
    output_path = OUTPUT_DIR / "metadata_test_predictions.csv"
    test_predictions.to_csv(output_path, index=False)
    print(f"\nSaved test predictions: {output_path}")
    print(f"Rows: {len(test_predictions)}")
    print(f"Columns: {list(test_predictions.columns)}")

    print(f"\nSaved models to: {MODEL_DIR}")
    for f in sorted(MODEL_DIR.glob("*.joblib")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
