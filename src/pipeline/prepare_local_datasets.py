"""
Prepare local datasets (local1.csv, local2.csv) for transfer learning and pseudo-labeling.

Outputs are written to processed_data/local/ as:
- local_all_raw.csv
- local_labeled_raw.csv
- local_unlabeled_raw.csv
- local_labeled_text_dataset.csv
- local_labeled_image_dataset.csv
- local_labeled_metadata_dataset.csv
- local_unlabeled_text_dataset.csv
- local_unlabeled_image_dataset.csv
- local_unlabeled_metadata_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]


EXPECTED_COLS = [
    "is_fraudulent",
    "product_title",
    "description",
    "review_1",
    "review1_rating",
    "review_2",
    "review2_rating",
    "listed_price",
    "original_price",
    "seller_rating",
    "rating_count",
    "item_rating",
    "item_rating_count",
    "image_url",
]


class TextPreprocessor:
    """Lightweight text preprocessor for local data prep (no external downloads)."""

    def __init__(self):
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "when", "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "as",
        }

    def clean_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.translate(str.maketrans({p: " " for p in string.punctuation}))
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [t for t in text.split() if t not in self.stop_words and len(t) > 1]
        return " ".join(tokens)


def parse_label(value):
    """Parse local fraud label to {0,1} or NaN for unlabeled."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    mapping = {
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0,
        "YES": 1,
        "NO": 0,
        "T": 1,
        "F": 0,
    }
    return mapping.get(s, np.nan)


def parse_numeric(value, default=np.nan):
    """Parse numeric values robustly from mixed string/number fields."""
    if pd.isna(value) or value is None:
        return default
    value_str = str(value).strip()
    if not value_str:
        return default
    value_str = re.sub(r"[$€£¥,]", "", value_str)
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return default


def is_valid_url(url):
    if pd.isna(url) or not str(url).strip():
        return False
    try:
        parsed = urlparse(str(url).strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_text_dataset(df: pd.DataFrame, tp: TextPreprocessor) -> pd.DataFrame:
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["fraud_label"] = df["fraud_label"]
    out["title_cleaned"] = df["product_title"].fillna("").apply(tp.clean_text)
    out["description_cleaned"] = df["description"].fillna("").apply(tp.clean_text)
    out["review1_cleaned"] = df["review_1"].fillna("").apply(tp.clean_text)
    out["review2_cleaned"] = df["review_2"].fillna("").apply(tp.clean_text)
    return out


def build_image_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["fraud_label"] = df["fraud_label"]
    out["image_url"] = df["image_url"]
    out = out[out["image_url"].apply(is_valid_url)].copy()
    return out


def build_metadata_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["fraud_label"] = df["fraud_label"]

    numeric_fields = [
        "listed_price",
        "original_price",
        "seller_rating",
        "rating_count",
        "item_rating",
        "item_rating_count",
        "review1_rating",
        "review2_rating",
    ]
    for col in numeric_fields:
        out[col] = df[col].apply(parse_numeric)

    for col in ["listed_price", "original_price"]:
        med = out[col].median()
        out[col] = out[col].fillna(0 if pd.isna(med) else med)

    for col in ["seller_rating", "item_rating", "review1_rating", "review2_rating"]:
        mean = out[col].mean()
        out[col] = out[col].fillna(0 if pd.isna(mean) else mean)

    for col in ["rating_count", "item_rating_count"]:
        med = out[col].median()
        out[col] = out[col].fillna(0 if pd.isna(med) else med)

    out["price_deviation"] = np.where(
        out["original_price"] > 0,
        ((out["original_price"] - out["listed_price"]) / out["original_price"]) * 100,
        0,
    )
    out["price_ratio"] = np.where(
        out["original_price"] > 0,
        out["listed_price"] / out["original_price"],
        1,
    )
    out["abnormal_discount"] = (out["price_deviation"] > 70).astype(int)
    out["review_rating_diff"] = (out["review1_rating"] - out["review2_rating"]).abs()
    out["seller_item_rating_gap"] = (out["seller_rating"] - out["item_rating"]).abs()

    scale_cols = [
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
    minmax = {}
    for col in scale_cols:
        min_v = float(out[col].min())
        max_v = float(out[col].max())
        minmax[col] = {"min": min_v, "max": max_v}
        if max_v > min_v:
            out[f"{col}_scaled"] = (out[col] - min_v) / (max_v - min_v)
        else:
            out[f"{col}_scaled"] = 0.0

    out.attrs["minmax"] = minmax
    return out


def align_modalities(text_df: pd.DataFrame, image_df: pd.DataFrame, meta_df: pd.DataFrame):
    common_ids = set(text_df["product_id"]) & set(image_df["product_id"]) & set(meta_df["product_id"])
    text_aligned = text_df[text_df["product_id"].isin(common_ids)].sort_values("product_id").reset_index(drop=True)
    image_aligned = image_df[image_df["product_id"].isin(common_ids)].sort_values("product_id").reset_index(drop=True)
    meta_aligned = meta_df[meta_df["product_id"].isin(common_ids)].sort_values("product_id").reset_index(drop=True)
    return text_aligned, image_aligned, meta_aligned


def prepare_split(df: pd.DataFrame, split_name: str, out_dir: Path, tp: TextPreprocessor):
    text_df = build_text_dataset(df, tp)
    image_df = build_image_dataset(df)
    meta_df = build_metadata_dataset(df)

    text_df, image_df, meta_df = align_modalities(text_df, image_df, meta_df)

    text_path = out_dir / f"local_{split_name}_text_dataset.csv"
    image_path = out_dir / f"local_{split_name}_image_dataset.csv"
    meta_path = out_dir / f"local_{split_name}_metadata_dataset.csv"
    minmax_path = out_dir / f"local_{split_name}_metadata_minmax.json"

    text_df.to_csv(text_path, index=False)
    image_df.to_csv(image_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    with open(minmax_path, "w", encoding="utf-8") as f:
        json.dump(meta_df.attrs.get("minmax", {}), f, indent=2)

    return {
        "split": split_name,
        "rows_input": int(len(df)),
        "rows_text": int(len(text_df)),
        "rows_image": int(len(image_df)),
        "rows_meta": int(len(meta_df)),
        "rows_aligned": int(len(text_df)),
        "text_path": str(text_path),
        "image_path": str(image_path),
        "meta_path": str(meta_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare local1/local2 for transfer learning and pseudo-labeling")
    parser.add_argument("--local1", type=str, default=str(REPO_DIR / "data" / "local" / "local1.csv"))
    parser.add_argument("--local2", type=str, default=str(REPO_DIR / "data" / "local" / "local2.csv"))
    parser.add_argument("--out", type=str, default="processed_data/local")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    local1 = ensure_columns(pd.read_csv(args.local1, encoding="utf-8"))
    local2 = ensure_columns(pd.read_csv(args.local2, encoding="utf-8"))

    local1 = local1.copy()
    local2 = local2.copy()
    local1["source"] = "local1"
    local2["source"] = "local2"

    local1["product_id"] = np.arange(900001, 900001 + len(local1))
    local2["product_id"] = np.arange(910001, 910001 + len(local2))

    combined = pd.concat([local1, local2], ignore_index=True)
    combined["fraud_label"] = combined["is_fraudulent"].apply(parse_label)

    labeled_df = combined[combined["fraud_label"].isin([0, 1])].copy()
    labeled_df["fraud_label"] = labeled_df["fraud_label"].astype(int)
    unlabeled_df = combined[combined["fraud_label"].isna()].copy()

    combined.to_csv(out_dir / "local_all_raw.csv", index=False)
    labeled_df.to_csv(out_dir / "local_labeled_raw.csv", index=False)
    unlabeled_df.to_csv(out_dir / "local_unlabeled_raw.csv", index=False)

    tp = TextPreprocessor()
    labeled_stats = prepare_split(labeled_df, "labeled", out_dir, tp)
    unlabeled_stats = prepare_split(unlabeled_df, "unlabeled", out_dir, tp)

    print("\n=== LOCAL DATA PREP SUMMARY ===")
    print(f"Combined rows:  {len(combined)}")
    print(f"Labeled rows:   {len(labeled_df)}")
    print(f"Unlabeled rows: {len(unlabeled_df)}")
    print("\nAligned modality rows:")
    print(f"  labeled:   {labeled_stats['rows_aligned']}")
    print(f"  unlabeled: {unlabeled_stats['rows_aligned']}")
    if len(labeled_df) > 0:
        print("\nLabeled class distribution:")
        print(labeled_df["fraud_label"].value_counts().sort_index().to_string())

    print("\nSaved files:")
    for p in sorted(out_dir.glob("*.csv")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
