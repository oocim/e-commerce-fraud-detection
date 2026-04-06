"""
Evaluate current global models on local labeled data.

Outputs:
  predictions/local/local_baseline_predictions.csv
  predictions/local/local_baseline_metrics.json
  predictions/local/experiment_log.csv
  predictions/local/methodology_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_DIR = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_DIR / "src" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

try:
    import predict as predict_module
    from predict import FraudDetector
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        "Missing Python dependency required by predict.py: "
        f"{missing}. Install project requirements first, e.g. `pip install -r requirements.txt`."
    ) from exc


LOCAL_DIR = (
    REPO_DIR / "processed_data" / "local"
    if (REPO_DIR / "processed_data" / "local").exists()
    else REPO_DIR / "data" / "local"
)
PRED_DIR = REPO_DIR / "predictions" / "local"
PRED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = LOCAL_DIR / "local_labeled_split_tags.csv"


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def get_split_tags(df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    """Create or load a fixed stratified split map for reproducible reporting."""
    if SPLIT_PATH.exists():
        split_df = pd.read_csv(SPLIT_PATH)
        needed = {"product_id", "fraud_label", "split"}
        if needed.issubset(split_df.columns):
            split_df["product_id"] = split_df["product_id"].astype(str)
            return split_df

    base = df[["product_id", "fraud_label"]].copy()
    base["product_id"] = base["product_id"].astype(str)

    train_ids, test_ids = train_test_split(
        base["product_id"].values,
        test_size=test_size,
        random_state=seed,
        stratify=base["fraud_label"].values,
    )
    train_ids = set(train_ids)
    split_df = base.copy()
    split_df["split"] = split_df["product_id"].apply(
        lambda x: "dev" if x in train_ids else "test"
    )
    split_df.to_csv(SPLIT_PATH, index=False)
    return split_df


def extract_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return normalized (product_id, fraud_label) rows with valid binary labels only."""
    if "fraud_label" not in df.columns:
        return pd.DataFrame(columns=["product_id", "fraud_label"])

    out = df[["product_id", "fraud_label"]].copy()
    out["product_id"] = out["product_id"].astype(str)
    out["fraud_label"] = pd.to_numeric(out["fraud_label"], errors="coerce")
    out = out[out["fraud_label"].isin([0, 1])].copy()
    out["fraud_label"] = out["fraud_label"].astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local baseline with reproducible split logging")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-timeout", type=float, default=5.0)
    args = parser.parse_args()

    text_path = LOCAL_DIR / "local_labeled_text_dataset.csv"
    image_path = LOCAL_DIR / "local_labeled_image_dataset.csv"
    meta_path = LOCAL_DIR / "local_labeled_metadata_dataset.csv"

    for p in [text_path, image_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    text_df = pd.read_csv(text_path)
    image_df = pd.read_csv(image_path)
    meta_df = pd.read_csv(meta_path)

    for d in [text_df, image_df, meta_df]:
        d["product_id"] = d["product_id"].astype(str)

    # Build labels from modality datasets only (no raw labeled file dependency).
    labels_pool = pd.concat(
        [extract_valid_labels(text_df), extract_valid_labels(image_df), extract_valid_labels(meta_df)],
        ignore_index=True,
    )
    if labels_pool.empty:
        raise ValueError(
            "No valid binary labels found in modality datasets. "
            "Ensure fraud_label exists and contains only 0/1 values in at least one modality CSV."
        )

    conflict_counts = labels_pool.groupby("product_id")["fraud_label"].nunique()
    conflicting_ids = conflict_counts[conflict_counts > 1]
    if len(conflicting_ids):
        raise ValueError(
            f"Found inconsistent fraud_label values across modality datasets for {len(conflicting_ids)} product_id rows"
        )

    label_df = labels_pool.drop_duplicates(subset=["product_id", "fraud_label"]).copy()

    split_tags = get_split_tags(label_df, test_size=args.test_size, seed=args.seed)
    eval_ids_df = split_tags.copy()
    if args.split != "all":
        eval_ids_df = eval_ids_df[eval_ids_df["split"] == args.split].copy()
    if len(eval_ids_df) == 0:
        raise ValueError(f"No rows available for split={args.split}")

    eval_ids = set(eval_ids_df["product_id"].tolist())

    # Subset each prepared modality dataset to evaluation IDs.
    text_eval = text_df[text_df["product_id"].isin(eval_ids)].copy()
    image_eval = image_df[image_df["product_id"].isin(eval_ids)].copy()
    meta_eval = meta_df[meta_df["product_id"].isin(eval_ids)].copy()

    # Build text input directly from cleaned fields (no raw re-cleaning needed).
    for col in ["title_cleaned", "description_cleaned", "review1_cleaned", "review2_cleaned"]:
        if col not in text_eval.columns:
            text_eval[col] = ""
        text_eval[col] = text_eval[col].fillna("")
    text_eval["text"] = (
        "Title: " + text_eval["title_cleaned"]
        + " Description: " + text_eval["description_cleaned"]
        + " Review1: " + text_eval["review1_cleaned"]
        + " Review2: " + text_eval["review2_cleaned"]
    )

    # Prevent runtime stalls from nltk.download in predict.TextPreprocessor.
    predict_module.NLTK_AVAILABLE = False

    # Speed up slow image URL inference for local evaluation.
    if hasattr(predict_module, "requests"):
        _orig_get = predict_module.requests.get

        def _fast_get(*a, **kw):
            kw["timeout"] = args.image_timeout
            return _orig_get(*a, **kw)

        predict_module.requests.get = _fast_get

    detector = FraudDetector()
    detector.load_models()

    ensemble_threshold = float(getattr(detector, "threshold", 0.515))
    print(f"Using ensemble threshold: {ensemble_threshold}")

    # Run per-modality predictions on prepared modality datasets.
    text_probs = detector.predict_text(text_eval) if len(text_eval) else np.array([])
    image_probs = detector.predict_image(image_eval) if len(image_eval) else np.array([])
    meta_probs = detector.predict_metadata(meta_eval) if len(meta_eval) else np.array([])

    text_pred = pd.DataFrame({"product_id": text_eval["product_id"].values, "text_fraud_proba": text_probs})
    image_pred = pd.DataFrame({"product_id": image_eval["product_id"].values, "image_fraud_proba": image_probs})
    meta_pred = pd.DataFrame({"product_id": meta_eval["product_id"].values, "metadata_fraud_proba": meta_probs})

    pred_df = eval_ids_df[["product_id", "fraud_label"]].copy()
    pred_df = pred_df.merge(text_pred, on="product_id", how="left")
    pred_df = pred_df.merge(image_pred, on="product_id", how="left")
    pred_df = pred_df.merge(meta_pred, on="product_id", how="left")

    # Weighted ensemble with row-wise NaN-safe fallback, same logic as predict.py.
    weights = np.array([detector.weights["text"], detector.weights["image"], detector.weights["metadata"]], dtype=float)
    probs_matrix = np.column_stack([
        pred_df["text_fraud_proba"].values,
        pred_df["image_fraud_proba"].values,
        pred_df["metadata_fraud_proba"].values,
    ])
    ensemble_probs = np.zeros(len(pred_df), dtype=float)
    for i in range(len(pred_df)):
        valid = ~np.isnan(probs_matrix[i])
        if valid.any():
            w = weights[valid]
            p = probs_matrix[i][valid]
            ensemble_probs[i] = float(np.dot(w, p) / w.sum())
        else:
            ensemble_probs[i] = 0.5

    pred_df["ensemble_fraud_proba"] = ensemble_probs
    pred_df["fraud_prediction"] = (pred_df["ensemble_fraud_proba"] >= ensemble_threshold).astype(int)
    pred_df["confidence"] = np.abs(pred_df["ensemble_fraud_proba"] - 0.5) * 2

    y_true = pred_df["fraud_label"].astype(int).values
    metrics = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "script": "evaluate_local_baseline.py",
        },
        "dataset": {
            "name": "local_labeled_modality_datasets",
            "split": args.split,
            "rows_input": int(len(eval_ids_df)),
            "rows_scored": int(len(pred_df)),
            "fraud_count": int(eval_ids_df["fraud_label"].sum()),
            "legit_count": int((eval_ids_df["fraud_label"] == 0).sum()),
            "rows_text_available": int(pred_df["text_fraud_proba"].notna().sum()),
            "rows_image_available": int(pred_df["image_fraud_proba"].notna().sum()),
            "rows_metadata_available": int(pred_df["metadata_fraud_proba"].notna().sum()),
        },
        "model_config": {
            "weights": detector.weights,
            "temperature": detector.temperature,
            "ensemble_threshold": ensemble_threshold,
            "loaded_modalities": detector._loaded,
        },
        "metrics": {},
    }

    # Per-modality metrics on the rows each modality successfully scored.
    modality_map = {
        "text": ("text_fraud_proba", 0.5),
        "image": ("image_fraud_proba", 0.5),
        "metadata": ("metadata_fraud_proba", 0.5),
        "ensemble": ("ensemble_fraud_proba", ensemble_threshold),
    }

    for name, (col, thr) in modality_map.items():
        valid = pred_df[col].notna()
        if valid.sum() == 0:
            metrics["metrics"][name] = {"samples": 0, "status": "no_valid_predictions"}
            continue

        y_true_sub = pred_df.loc[valid, "fraud_label"].astype(int).values
        y_prob_sub = pred_df.loc[valid, col].astype(float).values
        metrics["metrics"][name] = compute_metrics(y_true_sub, y_prob_sub, threshold=thr)

    # Identify strongest single modality by F1 for thesis baseline statement.
    single_modalities = [m for m in ["text", "image", "metadata"] if "f1" in metrics["metrics"].get(m, {})]
    if single_modalities:
        best_single = max(single_modalities, key=lambda m: metrics["metrics"][m]["f1"])
        metrics["best_single_modality"] = {
            "name": best_single,
            "f1": metrics["metrics"][best_single]["f1"],
            "roc_auc": metrics["metrics"][best_single]["roc_auc"],
        }

    # Save full predictions and metrics snapshot.
    pred_out = PRED_DIR / "local_baseline_predictions.csv"
    metrics_out = PRED_DIR / "local_baseline_metrics.json"
    method_out = PRED_DIR / "methodology_snapshot.json"
    pred_df.to_csv(pred_out, index=False)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    methodology = {
        "task": "local baseline evaluation before transfer learning",
        "data_sources": {
            "labeled_text": str(text_path),
            "labeled_image": str(image_path),
            "labeled_metadata": str(meta_path),
            "split_tags": str(SPLIT_PATH),
            "labels_origin": "derived from modality fraud_label columns with consistency checks",
            "prepared_by": "prepare_local_datasets.py",
        },
        "split_strategy": {
            "type": "stratified holdout",
            "test_size": args.test_size,
            "seed": args.seed,
            "evaluated_split": args.split,
        },
        "inference_artifacts": {
            "text": "saved_models/text/hybrid_roberta_tfidf.pth",
            "image": "saved_models/image/best_resnet50.pth",
            "metadata": "saved_models/metadata/*.joblib",
            "ensemble_config": "saved_models/ensemble_weights.json",
        },
        "decision_rules": {
            "text_threshold": 0.5,
            "image_threshold": 0.5,
            "metadata_threshold": 0.5,
            "ensemble_threshold": ensemble_threshold,
        },
        "notes": [
            "Per-modality metrics use prepared modality datasets (no raw URL dependence for text/metadata).",
            "Per-modality metrics use only rows where modality produced valid probability.",
            "Ensemble probability is weighted average with configured temperatures and weights.",
            "This run should be cited as pre-transfer local baseline in thesis results.",
        ],
    }
    with open(method_out, "w", encoding="utf-8") as f:
        json.dump(methodology, f, indent=2)

    # Append concise row to experiment log for paper-ready tracking.
    log_path = PRED_DIR / "experiment_log.csv"
    row = {
        "run_timestamp": metrics["run_timestamp"],
        "experiment": f"local_baseline_pre_transfer_{args.split}",
        "split": args.split,
        "test_size": args.test_size,
        "seed": args.seed,
        "rows_scored": metrics["dataset"]["rows_scored"],
        "fraud_count": metrics["dataset"]["fraud_count"],
        "text_f1": metrics["metrics"].get("text", {}).get("f1", np.nan),
        "text_auc": metrics["metrics"].get("text", {}).get("roc_auc", np.nan),
        "image_f1": metrics["metrics"].get("image", {}).get("f1", np.nan),
        "image_auc": metrics["metrics"].get("image", {}).get("roc_auc", np.nan),
        "metadata_f1": metrics["metrics"].get("metadata", {}).get("f1", np.nan),
        "metadata_auc": metrics["metrics"].get("metadata", {}).get("roc_auc", np.nan),
        "ensemble_f1": metrics["metrics"].get("ensemble", {}).get("f1", np.nan),
        "ensemble_auc": metrics["metrics"].get("ensemble", {}).get("roc_auc", np.nan),
        "ensemble_threshold": ensemble_threshold,
    }
    row_df = pd.DataFrame([row])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, index=False)

    print("\nSaved outputs:")
    print(f"  - {pred_out}")
    print(f"  - {metrics_out}")
    print(f"  - {method_out}")
    print(f"  - {log_path}")

    print("\nKey metrics:")
    for name in ["text", "image", "metadata", "ensemble"]:
        m = metrics["metrics"].get(name, {})
        if "f1" in m:
            print(f"  {name:8s}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  n={m['samples']}")
        else:
            print(f"  {name:8s}  {m.get('status', 'unavailable')}")


if __name__ == "__main__":
    main()
