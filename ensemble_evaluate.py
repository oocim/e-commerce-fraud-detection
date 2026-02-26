"""
Multimodal Ensemble Evaluation

Combines test-set predictions from all three modality models:
  1. Text   (RoBERTa + TF-IDF)  → text_test_predictions.csv
  2. Image  (ResNet-50)          → image_test_predictions.csv
  3. Metadata (XGBoost/RF/LR)    → metadata_test_predictions.csv

Fusion strategies:
  A. Simple average of fraud probabilities
  B. Weighted average (weights from per-model ROC-AUC)
  C. Majority vote on hard predictions
  D. Stacking meta-learner (Logistic Regression on probabilities)

Outputs final ensemble metrics and saves combined predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
PRED_DIR = PROJECT_DIR / "predictions"
OUTPUT_DIR = PROJECT_DIR / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42


def load_predictions():
    """Load and merge predictions from all three modalities."""
    text_path = PRED_DIR / "text_test_predictions.csv"
    image_path = PRED_DIR / "image_test_predictions.csv"
    meta_path = PRED_DIR / "metadata_test_predictions.csv"

    missing = [p for p in [text_path, image_path, meta_path] if not p.exists()]
    if missing:
        print("Missing prediction files:")
        for p in missing:
            print(f"  - {p}")
        print("\nAvailable files in predictions/:")
        if PRED_DIR.exists():
            for f in sorted(PRED_DIR.glob("*.csv")):
                print(f"  - {f.name}")
        raise FileNotFoundError("Run all three models first and place CSVs in predictions/")

    text_df = pd.read_csv(text_path)
    image_df = pd.read_csv(image_path)
    meta_df = pd.read_csv(meta_path)

    print(f"Text  predictions: {len(text_df)} rows")
    print(f"Image predictions: {len(image_df)} rows")
    print(f"Meta  predictions: {len(meta_df)} rows")

    # Ensure product_id is string for joining
    for df in [text_df, image_df, meta_df]:
        df["product_id"] = df["product_id"].astype(str).str.strip()

    # Merge on product_id (inner join — only products present in ALL modalities)
    merged = text_df[["product_id", "fraud_label", "text_fraud_proba"]].merge(
        image_df[["product_id", "image_fraud_proba"]], on="product_id", how="inner"
    ).merge(
        meta_df[["product_id", "metadata_fraud_proba"]], on="product_id", how="inner"
    )

    print(f"\nMerged (all 3 modalities): {len(merged)} rows")
    print(f"Fraud distribution:\n{merged['fraud_label'].value_counts()}\n")

    return merged


def evaluate(y_true, y_pred, y_proba, label=""):
    """Print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    print(f"{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"F1 Score (fraud)  : {f1:.4f}")
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"Avg Precision (PR): {ap:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")

    return {"accuracy": acc, "f1": f1, "roc_auc": roc, "avg_precision": ap}


def main():
    print("=" * 60)
    print("  MULTIMODAL ENSEMBLE EVALUATION")
    print("=" * 60 + "\n")

    merged = load_predictions()

    y_true = merged["fraud_label"].values
    p_text = merged["text_fraud_proba"].values
    p_image = merged["image_fraud_proba"].values
    p_meta = merged["metadata_fraud_proba"].values

    results = {}

    # ── Individual model baselines ───────────────────────────────
    for name, proba in [("Text (RoBERTa+TF-IDF)", p_text),
                        ("Image (ResNet-50)", p_image),
                        ("Metadata (Best ML)", p_meta)]:
        preds = (proba >= 0.5).astype(int)
        results[name] = evaluate(y_true, preds, proba, label=name)

    # ── Strategy A: Simple Average ───────────────────────────────
    avg_proba = (p_text + p_image + p_meta) / 3
    avg_preds = (avg_proba >= 0.5).astype(int)
    results["Ensemble: Simple Average"] = evaluate(
        y_true, avg_preds, avg_proba, label="Ensemble: Simple Average"
    )

    # ── Strategy B: Weighted Average (weights = per-model ROC-AUC) ─
    w_text = roc_auc_score(y_true, p_text)
    w_image = roc_auc_score(y_true, p_image)
    w_meta = roc_auc_score(y_true, p_meta)
    w_sum = w_text + w_image + w_meta

    weighted_proba = (w_text * p_text + w_image * p_image + w_meta * p_meta) / w_sum
    weighted_preds = (weighted_proba >= 0.5).astype(int)
    print(f"Weights (from ROC-AUC): text={w_text:.4f}, image={w_image:.4f}, meta={w_meta:.4f}\n")
    results["Ensemble: Weighted Average (AUC)"] = evaluate(
        y_true, weighted_preds, weighted_proba, label="Ensemble: Weighted Average (AUC)"
    )

    # ── Strategy B2: Tuned Weights (image down-weighted) ─────────
    tw_text, tw_image, tw_meta = 0.4, 0.2, 0.4
    tuned_proba = tw_text * p_text + tw_image * p_image + tw_meta * p_meta
    tuned_preds = (tuned_proba >= 0.5).astype(int)
    print(f"Tuned weights: text={tw_text}, image={tw_image}, meta={tw_meta}\n")
    results["Ensemble: Tuned Weights"] = evaluate(
        y_true, tuned_preds, tuned_proba, label="Ensemble: Tuned Weights"
    )

    # ── Strategy C: Majority Vote ────────────────────────────────
    votes = (
        (p_text >= 0.5).astype(int)
        + (p_image >= 0.5).astype(int)
        + (p_meta >= 0.5).astype(int)
    )
    majority_preds = (votes >= 2).astype(int)
    # Use average proba for ROC-AUC calculation
    results["Ensemble: Majority Vote"] = evaluate(
        y_true, majority_preds, avg_proba, label="Ensemble: Majority Vote"
    )

    # ── Strategy D: Stacking (Logistic Regression meta-learner) ──
    X_stack = np.column_stack([p_text, p_image, p_meta])
    meta_clf = LogisticRegression(random_state=SEED, max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_f1 = cross_val_score(meta_clf, X_stack, y_true, cv=cv, scoring="f1")
    print(f"Stacking 5-Fold CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})\n")

    meta_clf.fit(X_stack, y_true)
    stack_proba = meta_clf.predict_proba(X_stack)[:, 1]
    stack_preds = meta_clf.predict(X_stack)
    results["Ensemble: Stacking (LR)"] = evaluate(
        y_true, stack_preds, stack_proba, label="Ensemble: Stacking (LR)"
    )

    # ── Summary comparison ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ENSEMBLE COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    summary.index.name = "Strategy"
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary["f1"].idxmax()
    print(f"\n>>> Best by F1: {best} ({summary.loc[best, 'f1']:.4f})")

    # ── Save merged predictions ──────────────────────────────────
    merged["ensemble_avg_proba"] = avg_proba
    merged["ensemble_weighted_auc_proba"] = weighted_proba
    merged["ensemble_tuned_proba"] = tuned_proba
    merged["ensemble_majority_pred"] = majority_preds
    merged["ensemble_stack_proba"] = stack_proba
    merged["ensemble_final_pred"] = tuned_preds

    output_path = OUTPUT_DIR / "ensemble_predictions.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved ensemble predictions: {output_path}")


if __name__ == "__main__":
    main()
