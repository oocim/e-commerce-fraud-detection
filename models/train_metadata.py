"""
Train metadata fraud models and classify Lazada records.

This script:
1) Trains Random Forest, Logistic Regression, and XGBoost using
   metadata_dataset + synthetic_metadata_dataset
2) Uses all metadata features (raw, derived, scaled)
3) Applies the trained models to all rows in lazada_normalized_dedup.csv
   as an unlabeled test set (ignores is_fraudulent column)
4) Saves per-row predictions and model probability outputs
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
LAZADA_PATH = PROJECT_DIR / "processed_data" / "lazada_normalized_dedup.csv"
OUTPUT_DIR = PROJECT_DIR / "lazada_predictions"
MODEL_DIR = PROJECT_DIR / "saved_models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def _safe_numeric(series: pd.Series, fill_value: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def engineer_metadata_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Create metadata features compatible with metadata_dataset training schema."""
    output = df.copy()

    required_raw_cols = [
        "listed_price",
        "original_price",
        "seller_rating",
        "rating_count",
        "item_rating",
        "item_rating_count",
        "review1_rating",
        "review2_rating",
    ]

    for col in required_raw_cols:
        if col not in output.columns:
            output[col] = np.nan

    output["listed_price"] = _safe_numeric(output["listed_price"], ref_df["listed_price"].median())
    output["original_price"] = _safe_numeric(output["original_price"], ref_df["original_price"].median())
    output["seller_rating"] = _safe_numeric(output["seller_rating"], ref_df["seller_rating"].mean())
    output["rating_count"] = _safe_numeric(output["rating_count"], ref_df["rating_count"].median())
    output["item_rating"] = _safe_numeric(output["item_rating"], ref_df["item_rating"].mean())
    output["item_rating_count"] = _safe_numeric(output["item_rating_count"], ref_df["item_rating_count"].median())
    output["review1_rating"] = _safe_numeric(output["review1_rating"], ref_df["review1_rating"].mean())
    output["review2_rating"] = _safe_numeric(output["review2_rating"], ref_df["review2_rating"].mean())

    output["price_deviation"] = np.where(
        output["original_price"] > 0,
        ((output["original_price"] - output["listed_price"]) / output["original_price"]) * 100,
        0.0,
    )
    output["price_ratio"] = np.where(
        output["original_price"] > 0,
        output["listed_price"] / output["original_price"],
        1.0,
    )
    output["abnormal_discount"] = (output["price_deviation"] > 70).astype(int)
    output["review_rating_diff"] = (output["review1_rating"] - output["review2_rating"]).abs()
    output["seller_item_rating_gap"] = (output["seller_rating"] - output["item_rating"]).abs()

    features_to_scale = [
        "listed_price",
        "original_price",
        "price_deviation",
        "price_ratio",
        "seller_rating",
        "rating_count",
        "item_rating",
        "item_rating_count",
        "review1_rating",
        "review2_rating",
        "review_rating_diff",
        "seller_item_rating_gap",
    ]

    for col in features_to_scale:
        min_val = ref_df[col].min()
        max_val = ref_df[col].max()
        if max_val > min_val:
            output[f"{col}_scaled"] = (output[col] - min_val) / (max_val - min_val)
        else:
            output[f"{col}_scaled"] = 0.0

    return output


def main() -> None:
    print("Loading training datasets...")
    metadata_df = pd.read_csv(DATA_PATH)
    print(f"Fraud distribution:\n{metadata_df['fraud_label'].value_counts()}\n")

    synth_path = next((path for path in SYNTH_PATH_CANDIDATES if path.exists()), None)
    if synth_path is None:
        raise FileNotFoundError("Synthetic metadata dataset not found.")

    synthetic_df = pd.read_csv(synth_path)
    print(f"Synthetic dataset loaded from: {synth_path}")
    print(f"Synthetic shape: {synthetic_df.shape}")

    train_df = pd.concat([metadata_df, synthetic_df], ignore_index=True)
    print(f"Combined dataset shape: {train_df.shape}")
    print(f"Combined fraud distribution:\n{train_df['fraud_label'].value_counts()}\n")

    feature_cols = [c for c in metadata_df.columns if c not in ["product_id", "fraud_label"]]

    X = train_df[feature_cols].copy()
    y = train_df["fraud_label"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"Train fraud ratio: {y_train.mean():.4f}")
    print(f"Test  fraud ratio: {y_test.mean():.4f}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    print("Training and evaluating models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary.index.name = "Model"
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary["f1"].idxmax()
    print(f"\n>>> Best model by F1 score: {best} ({summary.loc[best, 'f1']:.4f})\n")

    print("Training models on full metadata+synthetic dataset for Lazada inference...")
    trained_models = {}
    for name, model in models.items():
        full_scaler = StandardScaler().fit(X)
        X_scaled = full_scaler.transform(X)

        if name == "Logistic Regression":
            model.fit(X_scaled, y)
        else:
            model.fit(X, y)
        trained_models[name] = model

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}_lazada.joblib")

    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, MODEL_DIR / "scaler_lazada.joblib")

    print("Preparing Lazada test set (unlabeled inference)...")
    lazada_df = pd.read_csv(LAZADA_PATH)
    lazada_df = lazada_df.copy()

    if "product_id" not in lazada_df.columns:
        lazada_df["product_id"] = [f"lazada_{i+1:05d}" for i in range(len(lazada_df))]

    # Ignore label-like column in Lazada by design
    if "is_fraudulent" in lazada_df.columns:
        lazada_df = lazada_df.drop(columns=["is_fraudulent"])

    lazada_features_df = engineer_metadata_features(lazada_df, train_df)
    X_lazada = lazada_features_df[feature_cols].copy()
    X_lazada_scaled = scaler.transform(X_lazada)

    output = lazada_df.copy()

    print(f"Lazada rows to classify: {len(output)}")
    for name, model in trained_models.items():
        key = name.lower().replace(" ", "_")
        if name == "Logistic Regression":
            pred = model.predict(X_lazada_scaled)
            proba = model.predict_proba(X_lazada_scaled)[:, 1]
        else:
            pred = model.predict(X_lazada)
            proba = model.predict_proba(X_lazada)[:, 1]

        output[f"{key}_pred"] = pred
        output[f"{key}_fraud_proba"] = proba

        print(
            f"{name:20s} -> predicted fraud: {int(pred.sum())}/{len(pred)} "
            f"({pred.mean():.2%}), avg proba={proba.mean():.4f}"
        )

    vote_cols = [
        "random_forest_pred",
        "logistic_regression_pred",
        "xgboost_pred",
    ]
    output["fraud_votes"] = output[vote_cols].sum(axis=1)
    output["fraud_majority_pred"] = (output["fraud_votes"] >= 2).astype(int)

    proba_cols = [
        "random_forest_fraud_proba",
        "logistic_regression_fraud_proba",
        "xgboost_fraud_proba",
    ]
    output["fraud_avg_proba"] = output[proba_cols].mean(axis=1)

    output_path = OUTPUT_DIR / "lazada_metadata_predictions.csv"
    output.to_csv(output_path, index=False)

    print("\nSaved predictions:")
    print(output_path)
    print("\nMajority-vote prediction count:")
    print(output["fraud_majority_pred"].value_counts())


if __name__ == "__main__":
    main()
