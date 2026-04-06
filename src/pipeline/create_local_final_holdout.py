"""Create untouched local final-test holdout CSVs from modality-labeled datasets.

This script derives labels from modality datasets (no raw labeled dependency),
reuses or creates a reproducible split map, and exports the test split as
untouched holdout files for final evaluation.

Outputs (default in processed_data/local/final_holdout):
  final_holdout_ids.csv
  local_final_holdout_text_dataset.csv
  local_final_holdout_image_dataset.csv
  local_final_holdout_metadata_dataset.csv
  final_holdout_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_DIR = Path(__file__).resolve().parents[2]
LOCAL_DIR = (
    REPO_DIR / "processed_data" / "local"
    if (REPO_DIR / "processed_data" / "local").exists()
    else REPO_DIR / "data" / "local"
)
SPLIT_PATH = LOCAL_DIR / "local_labeled_split_tags.csv"


def extract_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "fraud_label" not in df.columns:
        return pd.DataFrame(columns=["product_id", "fraud_label"])

    out = df[["product_id", "fraud_label"]].copy()
    out["product_id"] = out["product_id"].astype(str)
    out["fraud_label"] = pd.to_numeric(out["fraud_label"], errors="coerce")
    out = out[out["fraud_label"].isin([0, 1])].copy()
    out["fraud_label"] = out["fraud_label"].astype(int)
    return out


def get_or_create_split_tags(label_df: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    if SPLIT_PATH.exists():
        split_df = pd.read_csv(SPLIT_PATH)
        needed = {"product_id", "fraud_label", "split"}
        if needed.issubset(split_df.columns):
            split_df["product_id"] = split_df["product_id"].astype(str)
            return split_df

    base = label_df[["product_id", "fraud_label"]].copy()
    train_ids, test_ids = train_test_split(
        base["product_id"].values,
        test_size=test_size,
        random_state=seed,
        stratify=base["fraud_label"].values,
    )
    train_ids = set(train_ids)
    split_df = base.copy()
    split_df["split"] = split_df["product_id"].apply(lambda x: "dev" if x in train_ids else "test")
    split_df.to_csv(SPLIT_PATH, index=False)
    return split_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create untouched final holdout CSVs for local evaluation")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DIR / "final_holdout"))
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

    labels_pool = pd.concat(
        [extract_valid_labels(text_df), extract_valid_labels(image_df), extract_valid_labels(meta_df)],
        ignore_index=True,
    )
    if labels_pool.empty:
        raise ValueError("No valid binary fraud_label values found in modality datasets")

    conflict_counts = labels_pool.groupby("product_id")["fraud_label"].nunique()
    conflicting_ids = conflict_counts[conflict_counts > 1]
    if len(conflicting_ids):
        raise ValueError(
            f"Found inconsistent fraud_label across modality datasets for {len(conflicting_ids)} product_id rows"
        )

    label_df = labels_pool.drop_duplicates(subset=["product_id", "fraud_label"]).copy()
    split_tags = get_or_create_split_tags(label_df, test_size=args.test_size, seed=args.seed)

    holdout_ids = split_tags[split_tags["split"] == "test"][["product_id", "fraud_label"]].copy()
    holdout_id_set = set(holdout_ids["product_id"].tolist())

    holdout_text = text_df[text_df["product_id"].isin(holdout_id_set)].copy()
    holdout_image = image_df[image_df["product_id"].isin(holdout_id_set)].copy()
    holdout_meta = meta_df[meta_df["product_id"].isin(holdout_id_set)].copy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_out = out_dir / "final_holdout_ids.csv"
    text_out = out_dir / "local_final_holdout_text_dataset.csv"
    image_out = out_dir / "local_final_holdout_image_dataset.csv"
    meta_out = out_dir / "local_final_holdout_metadata_dataset.csv"
    summary_out = out_dir / "final_holdout_summary.json"

    holdout_ids.to_csv(ids_out, index=False)
    holdout_text.to_csv(text_out, index=False)
    holdout_image.to_csv(image_out, index=False)
    holdout_meta.to_csv(meta_out, index=False)

    summary = {
        "split_path": str(SPLIT_PATH),
        "config": {"test_size": args.test_size, "seed": args.seed},
        "counts": {
            "holdout_ids": int(len(holdout_ids)),
            "holdout_fraud": int((holdout_ids["fraud_label"] == 1).sum()),
            "holdout_legit": int((holdout_ids["fraud_label"] == 0).sum()),
            "holdout_text_rows": int(len(holdout_text)),
            "holdout_image_rows": int(len(holdout_image)),
            "holdout_metadata_rows": int(len(holdout_meta)),
        },
        "outputs": {
            "ids": str(ids_out),
            "text": str(text_out),
            "image": str(image_out),
            "metadata": str(meta_out),
        },
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Created untouched final holdout files:")
    print(f"  - {ids_out}")
    print(f"  - {text_out}")
    print(f"  - {image_out}")
    print(f"  - {meta_out}")
    print(f"  - {summary_out}")


if __name__ == "__main__":
    main()
