"""Generate pseudo labels for local unlabeled data using the current ensemble.

Inputs (prepared modality datasets):
  processed_data/local/local_unlabeled_text_dataset.csv
  processed_data/local/local_unlabeled_image_dataset.csv
  processed_data/local/local_unlabeled_metadata_dataset.csv
  processed_data/local/local_unlabeled_raw.csv

Outputs:
  predictions/local/local_unlabeled_pseudo_all.csv
  predictions/local/local_unlabeled_pseudo_filtered.csv
  predictions/local/local_unlabeled_pseudo_stats.json
  processed_data/local/local_pseudo_labeled_raw.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_text_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["title_cleaned", "description_cleaned", "review1_cleaned", "review2_cleaned"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    out["text"] = (
        "Title: " + out["title_cleaned"]
        + " Description: " + out["description_cleaned"]
        + " Review1: " + out["review1_cleaned"]
        + " Review2: " + out["review2_cleaned"]
    )
    return out


def sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local pseudo labels for transfer learning")
    parser.add_argument("--run-id", type=str, default="", help="Optional run identifier for tracking")
    parser.add_argument("--round", type=int, default=1, help="Pseudo-labeling round index")
    parser.add_argument("--fraud-min", type=float, default=0.90, help="Min ensemble prob for pseudo-fraud")
    parser.add_argument("--legit-max", type=float, default=0.10, help="Max ensemble prob for pseudo-legit")
    parser.add_argument("--agreement", type=int, default=2, choices=[2, 3], help="Required modality agreement count")
    parser.add_argument("--image-timeout", type=float, default=5.0, help="Image request timeout in seconds")
    parser.add_argument("--max-legit-ratio", type=float, default=3.0, help="Max pseudo-legit / pseudo-fraud ratio")
    args = parser.parse_args()

    raw_path = LOCAL_DIR / "local_unlabeled_raw.csv"
    text_path = LOCAL_DIR / "local_unlabeled_text_dataset.csv"
    image_path = LOCAL_DIR / "local_unlabeled_image_dataset.csv"
    meta_path = LOCAL_DIR / "local_unlabeled_metadata_dataset.csv"
    for p in [raw_path, text_path, image_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    raw_df = pd.read_csv(raw_path)
    text_df = pd.read_csv(text_path)
    image_df = pd.read_csv(image_path)
    meta_df = pd.read_csv(meta_path)
    for d in [raw_df, text_df, image_df, meta_df]:
        d["product_id"] = d["product_id"].astype(str)

    # Prevent runtime stalls from nltk downloads during inference.
    predict_module.NLTK_AVAILABLE = False
    if hasattr(predict_module, "requests"):
        _orig_get = predict_module.requests.get

        def _fast_get(*a, **kw):
            kw["timeout"] = args.image_timeout
            return _orig_get(*a, **kw)

        predict_module.requests.get = _fast_get

    detector = FraudDetector()
    detector.load_models()

    text_eval = build_text_input(text_df)
    image_eval = image_df.copy()
    meta_eval = meta_df.copy()

    text_probs = detector.predict_text(text_eval) if len(text_eval) else np.array([])
    image_probs = detector.predict_image(image_eval) if len(image_eval) else np.array([])
    meta_probs = detector.predict_metadata(meta_eval) if len(meta_eval) else np.array([])

    text_pred = pd.DataFrame({"product_id": text_eval["product_id"], "text_fraud_proba": text_probs})
    image_pred = pd.DataFrame({"product_id": image_eval["product_id"], "image_fraud_proba": image_probs})
    meta_pred = pd.DataFrame({"product_id": meta_eval["product_id"], "metadata_fraud_proba": meta_probs})

    all_ids = pd.DataFrame({"product_id": sorted(set(raw_df["product_id"]))})
    pred_df = all_ids.merge(text_pred, on="product_id", how="left")
    pred_df = pred_df.merge(image_pred, on="product_id", how="left")
    pred_df = pred_df.merge(meta_pred, on="product_id", how="left")

    weights = np.array([
        detector.weights.get("text", 0.0),
        detector.weights.get("image", 0.0),
        detector.weights.get("metadata", 0.0),
    ], dtype=float)
    probs_matrix = np.column_stack([
        pred_df["text_fraud_proba"].values,
        pred_df["image_fraud_proba"].values,
        pred_df["metadata_fraud_proba"].values,
    ])

    ensemble_probs = np.zeros(len(pred_df), dtype=float)
    modal_votes = np.zeros((len(pred_df), 3), dtype=float)
    for i in range(len(pred_df)):
        valid = ~np.isnan(probs_matrix[i])
        if valid.any():
            w = weights[valid]
            p = probs_matrix[i][valid]
            ensemble_probs[i] = float(np.dot(w, p) / w.sum())
        else:
            ensemble_probs[i] = 0.5
        modal_votes[i, 0] = 1 if (not np.isnan(probs_matrix[i, 0]) and probs_matrix[i, 0] >= 0.5) else 0
        modal_votes[i, 1] = 1 if (not np.isnan(probs_matrix[i, 1]) and probs_matrix[i, 1] >= 0.5) else 0
        modal_votes[i, 2] = 1 if (not np.isnan(probs_matrix[i, 2]) and probs_matrix[i, 2] >= 0.5) else 0

    pred_df["ensemble_fraud_proba"] = ensemble_probs
    pred_df["ensemble_pred"] = (pred_df["ensemble_fraud_proba"] >= 0.5).astype(int)
    pred_df["confidence"] = np.maximum(pred_df["ensemble_fraud_proba"], 1 - pred_df["ensemble_fraud_proba"])
    pred_df["modal_agree_count"] = np.where(
        pred_df["ensemble_pred"].values == 1,
        modal_votes.sum(axis=1),
        (1 - modal_votes).sum(axis=1),
    ).astype(int)

    # Strict confidence + agreement filter.
    high_fraud = pred_df["ensemble_fraud_proba"] >= args.fraud_min
    high_legit = pred_df["ensemble_fraud_proba"] <= args.legit_max
    confidence_pass = high_fraud | high_legit
    agreement_pass = pred_df["modal_agree_count"] >= args.agreement
    filtered = pred_df[confidence_pass & agreement_pass].copy()
    filtered["pseudo_label"] = np.where(filtered["ensemble_fraud_proba"] >= 0.5, 1, 0)

    # Class-ratio cap: avoid overwhelming with easy pseudo-legit.
    n_fraud = int((filtered["pseudo_label"] == 1).sum())
    n_legit = int((filtered["pseudo_label"] == 0).sum())
    if n_fraud > 0 and n_legit > int(args.max_legit_ratio * n_fraud):
        keep_legit = int(args.max_legit_ratio * n_fraud)
        legit_df = filtered[filtered["pseudo_label"] == 0].sort_values("confidence", ascending=False).head(keep_legit)
        fraud_df = filtered[filtered["pseudo_label"] == 1]
        filtered = pd.concat([fraud_df, legit_df], ignore_index=True)
        filtered = filtered.sort_values("confidence", ascending=False).reset_index(drop=True)

    # Join back to raw unlabeled rows for training convenience.
    pseudo_raw = raw_df.merge(
        filtered[["product_id", "pseudo_label", "ensemble_fraud_proba", "confidence", "modal_agree_count"]],
        on="product_id",
        how="inner",
    ).copy()
    pseudo_raw = pseudo_raw.rename(columns={"pseudo_label": "fraud_label"})

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    run_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"pseudo_r{args.round}_{run_token}"
    run_id_clean = sanitize_run_id(run_id)
    run_tag = f"{run_token}_{run_id_clean}_r{args.round}"

    # Canonical latest outputs (kept for downstream scripts)
    all_out = PRED_DIR / "local_unlabeled_pseudo_all.csv"
    filt_out = PRED_DIR / "local_unlabeled_pseudo_filtered.csv"
    stats_out = PRED_DIR / "local_unlabeled_pseudo_stats.json"
    pseudo_raw_out = LOCAL_DIR / "local_pseudo_labeled_raw.csv"

    # Archived outputs (never overwritten)
    pred_runs_dir = PRED_DIR / "pseudo_runs"
    local_runs_dir = LOCAL_DIR / "pseudo_runs"
    pred_runs_dir.mkdir(parents=True, exist_ok=True)
    local_runs_dir.mkdir(parents=True, exist_ok=True)
    all_out_run = pred_runs_dir / f"{run_tag}_all.csv"
    filt_out_run = pred_runs_dir / f"{run_tag}_filtered.csv"
    stats_out_run = pred_runs_dir / f"{run_tag}_stats.json"
    pseudo_raw_out_run = local_runs_dir / f"{run_tag}_pseudo_raw.csv"

    pred_df.to_csv(all_out, index=False)
    filtered.to_csv(filt_out, index=False)
    pseudo_raw.to_csv(pseudo_raw_out, index=False)
    pred_df.to_csv(all_out_run, index=False)
    filtered.to_csv(filt_out_run, index=False)
    pseudo_raw.to_csv(pseudo_raw_out_run, index=False)

    stats = {
        "run_timestamp": run_timestamp,
        "run_id": run_id,
        "round": args.round,
        "inputs": {
            "raw": str(raw_path),
            "text": str(text_path),
            "image": str(image_path),
            "metadata": str(meta_path),
        },
        "config": {
            "fraud_min": args.fraud_min,
            "legit_max": args.legit_max,
            "agreement": args.agreement,
            "max_legit_ratio": args.max_legit_ratio,
            "weights": detector.weights,
            "temperature": detector.temperature,
        },
        "counts": {
            "unlabeled_total": int(len(pred_df)),
            "image_available": int(pred_df["image_fraud_proba"].notna().sum()),
            "text_available": int(pred_df["text_fraud_proba"].notna().sum()),
            "metadata_available": int(pred_df["metadata_fraud_proba"].notna().sum()),
            "filtered_total": int(len(filtered)),
            "filtered_fraud": int((filtered.get("pseudo_label", pd.Series(dtype=int)) == 1).sum()),
            "filtered_legit": int((filtered.get("pseudo_label", pd.Series(dtype=int)) == 0).sum()),
            "pseudo_raw_rows": int(len(pseudo_raw)),
        },
        "outputs": {
            "all_predictions": str(all_out),
            "filtered_predictions": str(filt_out),
            "pseudo_raw": str(pseudo_raw_out),
            "all_predictions_archived": str(all_out_run),
            "filtered_predictions_archived": str(filt_out_run),
            "pseudo_raw_archived": str(pseudo_raw_out_run),
        },
    }
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(stats_out_run, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Append compact row to the shared experiment log.
    exp_log_path = PRED_DIR / "experiment_log.csv"
    exp_row = {
        "run_timestamp": stats["run_timestamp"],
        "experiment": f"pseudo_label_round_{args.round}",
        "run_id": stats["run_id"],
        "split": "unlabeled",
        "test_size": np.nan,
        "seed": np.nan,
        "rows_scored": stats["counts"]["unlabeled_total"],
        "fraud_count": np.nan,
        "text_f1": np.nan,
        "text_auc": np.nan,
        "image_f1": np.nan,
        "image_auc": np.nan,
        "metadata_f1": np.nan,
        "metadata_auc": np.nan,
        "ensemble_f1": np.nan,
        "ensemble_auc": np.nan,
        "ensemble_threshold": np.nan,
        "pseudo_fraud_min": args.fraud_min,
        "pseudo_legit_max": args.legit_max,
        "pseudo_agreement": args.agreement,
        "pseudo_max_legit_ratio": args.max_legit_ratio,
        "pseudo_kept_total": stats["counts"]["filtered_total"],
        "pseudo_kept_fraud": stats["counts"]["filtered_fraud"],
        "pseudo_kept_legit": stats["counts"]["filtered_legit"],
    }
    exp_df = pd.DataFrame([exp_row])
    if exp_log_path.exists():
        existing = pd.read_csv(exp_log_path)
        all_cols = list(dict.fromkeys(list(existing.columns) + list(exp_df.columns)))
        existing = existing.reindex(columns=all_cols)
        exp_df = exp_df.reindex(columns=all_cols)
        out_df = pd.concat([existing, exp_df], ignore_index=True)
        out_df.to_csv(exp_log_path, index=False)
    else:
        exp_df.to_csv(exp_log_path, index=False)

    print("\nPseudo-label generation complete")
    print(f"Total unlabeled: {len(pred_df)}")
    print(f"Filtered kept: {len(filtered)}")
    if len(filtered):
        print(f"  Fraud: {(filtered['pseudo_label'] == 1).sum()} | Legit: {(filtered['pseudo_label'] == 0).sum()}")
    print(f"Saved: {all_out}")
    print(f"Saved: {filt_out}")
    print(f"Saved: {stats_out}")
    print(f"Saved: {pseudo_raw_out}")
    print(f"Archived: {all_out_run}")
    print(f"Archived: {filt_out_run}")
    print(f"Archived: {stats_out_run}")
    print(f"Archived: {pseudo_raw_out_run}")
    print(f"Saved: {exp_log_path}")


if __name__ == "__main__":
    main()
