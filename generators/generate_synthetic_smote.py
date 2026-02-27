"""
Generate Synthetic Fraud Data via SMOTE

This script uses SMOTE (Synthetic Minority Oversampling Technique) to generate
new fraud examples by interpolating in feature space.

Two strategies:
1. Metadata SMOTE:  Standard SMOTE on numerical metadata features → new rows
2. Text SMOTE:      SMOTE on TF-IDF vectors + nearest-neighbor text retrieval
                    (pairs each SMOTE point with the real fraud text closest
                     to it in TF-IDF space)

Usage:
    python generate_synthetic_smote.py
    python generate_synthetic_smote.py --n-generate 150 --strategy both

Output:
    processed_data/synthetic_metadata_dataset.csv
    processed_data/synthetic_text_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"
SEED = 42


# ═══════════════════════════════════════════════════════════════════
#  METADATA SMOTE
# ═══════════════════════════════════════════════════════════════════

def smote_metadata(n_generate=150):
    """
    Apply SMOTE on metadata features to generate synthetic fraud rows.

    Approach:
    - Load metadata_dataset.csv (real data, ~4.3% fraud)
    - SMOTE generates new fraud samples by interpolating between real fraud neighbors
    - Post-process to ensure realistic value ranges
    - Recompute derived features (price_deviation, etc.)
    - Apply min-max scaling consistent with real data
    """
    print("=" * 60)
    print("  METADATA SMOTE")
    print("=" * 60)

    meta_df = pd.read_csv(PROCESSED_DIR / "metadata_dataset.csv")
    print(f"Original dataset: {meta_df.shape}")
    print(f"Fraud distribution:\n{meta_df['fraud_label'].value_counts()}\n")

    n_fraud_orig = (meta_df["fraud_label"] == 1).sum()
    n_legit = (meta_df["fraud_label"] == 0).sum()

    # Target: original fraud + n_generate new fraud rows
    target_fraud = n_fraud_orig + n_generate

    # Base features for SMOTE (raw numeric, not derived/scaled)
    base_features = [
        "listed_price", "original_price", "seller_rating", "rating_count",
        "item_rating", "item_rating_count", "review1_rating", "review2_rating",
    ]

    X = meta_df[base_features].values
    y = meta_df["fraud_label"].values

    print(f"Applying SMOTE to generate {n_generate} new fraud rows...")
    print(f"Target distribution: legit={n_legit}, fraud={target_fraud}")

    smote = SMOTE(
        sampling_strategy={1: target_fraud, 0: n_legit},
        random_state=SEED,
        k_neighbors=min(5, n_fraud_orig - 1),  # k can't exceed minority count
    )
    X_res, y_res = smote.fit_resample(X, y)

    # Extract only the NEW synthetic rows (they're appended at the end)
    n_original = len(meta_df)
    X_new = X_res[n_original:]
    y_new = y_res[n_original:]

    print(f"Generated {len(X_new)} synthetic fraud rows")

    # Build DataFrame for new rows
    synth_df = pd.DataFrame(X_new, columns=base_features)
    synth_df["fraud_label"] = 1

    # Post-process to ensure realistic ranges
    synth_df["listed_price"] = synth_df["listed_price"].clip(lower=1.0).round(2)
    synth_df["original_price"] = synth_df["original_price"].clip(lower=1.0).round(2)
    synth_df["seller_rating"] = synth_df["seller_rating"].clip(0.0, 5.0).round(1)
    synth_df["rating_count"] = synth_df["rating_count"].clip(lower=0).round(0).astype(int)
    synth_df["item_rating"] = synth_df["item_rating"].clip(0.0, 5.0).round(1)
    synth_df["item_rating_count"] = synth_df["item_rating_count"].clip(lower=0).round(0).astype(int)
    synth_df["review1_rating"] = synth_df["review1_rating"].clip(1, 5).round(0).astype(int)
    synth_df["review2_rating"] = synth_df["review2_rating"].clip(1, 5).round(0).astype(int)

    # Recompute derived features (same logic as preprocessing_pipeline.py)
    synth_df["price_deviation"] = np.where(
        synth_df["original_price"] > 0,
        ((synth_df["original_price"] - synth_df["listed_price"]) / synth_df["original_price"]) * 100,
        0,
    )
    synth_df["price_ratio"] = np.where(
        synth_df["original_price"] > 0,
        synth_df["listed_price"] / synth_df["original_price"],
        1,
    )
    synth_df["abnormal_discount"] = (synth_df["price_deviation"] > 70).astype(int)
    synth_df["review_rating_diff"] = abs(synth_df["review1_rating"] - synth_df["review2_rating"])
    synth_df["seller_item_rating_gap"] = abs(synth_df["seller_rating"] - synth_df["item_rating"])

    # Apply min-max scaling using REAL dataset stats (not synthetic)
    features_to_scale = [
        "listed_price", "original_price", "price_deviation", "price_ratio",
        "seller_rating", "rating_count", "item_rating", "item_rating_count",
        "review1_rating", "review2_rating", "review_rating_diff", "seller_item_rating_gap",
    ]
    for col in features_to_scale:
        min_val = meta_df[col].min()
        max_val = meta_df[col].max()
        if max_val > min_val:
            synth_df[f"{col}_scaled"] = (synth_df[col] - min_val) / (max_val - min_val)
        else:
            synth_df[f"{col}_scaled"] = 0

    # Assign product IDs
    start_id = 30000
    synth_df.insert(0, "product_id", [str(start_id + i) for i in range(len(synth_df))])

    # Reorder columns to match real metadata_dataset.csv
    col_order = meta_df.columns.tolist()
    # Only keep columns that exist in both
    col_order = [c for c in col_order if c in synth_df.columns]
    synth_df = synth_df[col_order]

    # Save
    out_path = PROCESSED_DIR / "synthetic_metadata_dataset.csv"
    synth_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(synth_df)} rows)")

    # Quality summary
    print(f"\n── Quality Summary ──")
    print(f"Price range:       ${synth_df['listed_price'].min():.2f} - ${synth_df['listed_price'].max():.2f}")
    print(f"Seller rating:     {synth_df['seller_rating'].min():.1f} - {synth_df['seller_rating'].max():.1f}")
    print(f"Item rating:       {synth_df['item_rating'].min():.1f} - {synth_df['item_rating'].max():.1f}")
    print(f"Price deviation:   {synth_df['price_deviation'].min():.1f}% - {synth_df['price_deviation'].max():.1f}%")
    print(f"Abnormal discounts: {synth_df['abnormal_discount'].sum()} / {len(synth_df)}")

    return synth_df


# ═══════════════════════════════════════════════════════════════════
#  TEXT SMOTE (TF-IDF space + nearest-neighbor text retrieval)
# ═══════════════════════════════════════════════════════════════════

def smote_text(n_generate=150, smote_meta_df=None):
    """
    Apply SMOTE on TF-IDF features, then retrieve nearest real fraud text.

    Approach:
    - Fit TF-IDF on all text data
    - SMOTE generates synthetic TF-IDF vectors for the fraud class
    - For each synthetic vector, find the nearest REAL fraud example in TF-IDF space
    - Use that real example's text (optionally with small perturbations)
    - This gives us realistic text paired with novel feature-space points

    The key insight: SMOTE creates interpolated points between real fraud samples,
    so each synthetic point is "between" two real frauds. We retrieve the closest
    real fraud text so the text is always high-quality and readable.

    Optional: If smote_meta_df is provided, pair text rows with SMOTE metadata
    (for consistent product_ids across both modalities).
    """
    print("\n" + "=" * 60)
    print("  TEXT SMOTE (TF-IDF + nearest-neighbor retrieval)")
    print("=" * 60)

    text_df = pd.read_csv(PROCESSED_DIR / "text_dataset.csv")
    print(f"Original dataset: {text_df.shape}")
    print(f"Fraud distribution:\n{text_df['fraud_label'].value_counts()}\n")

    n_fraud_orig = (text_df["fraud_label"] == 1).sum()
    n_legit = (text_df["fraud_label"] == 0).sum()
    target_fraud = n_fraud_orig + n_generate

    # Build combined text field
    text_cols = ["title_cleaned", "description_cleaned", "review1_cleaned", "review2_cleaned"]
    for col in text_cols:
        text_df[col] = text_df[col].fillna("")
    text_df["combined_text"] = (
        text_df["title_cleaned"] + " " +
        text_df["description_cleaned"] + " " +
        text_df["review1_cleaned"] + " " +
        text_df["review2_cleaned"]
    )

    # Fit TF-IDF on all text
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(text_df["combined_text"]).toarray()
    y = text_df["fraud_label"].values

    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    print(f"Applying SMOTE in TF-IDF space...")

    smote = SMOTE(
        sampling_strategy={1: target_fraud, 0: n_legit},
        random_state=SEED,
        k_neighbors=min(5, n_fraud_orig - 1),
    )
    X_res, y_res = smote.fit_resample(X_tfidf, y)

    # Extract synthetic fraud vectors
    X_synth = X_res[len(text_df):]
    print(f"Generated {len(X_synth)} synthetic TF-IDF vectors")

    # Find nearest real fraud for each synthetic point
    fraud_mask = text_df["fraud_label"] == 1
    fraud_indices = np.where(fraud_mask.values)[0]
    X_fraud_real = X_tfidf[fraud_indices]
    fraud_texts = text_df.loc[fraud_mask].reset_index(drop=True)

    print("Finding nearest real fraud neighbors...")
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X_fraud_real)
    distances, neighbor_idx = nn.kneighbors(X_synth)

    # For each synthetic point, pick a neighbor (alternate between k=1 and k=2
    # to avoid always using the same nearest example)
    synth_text_rows = []
    start_id = 30000
    neighbor_usage = {}  # track how often each real fraud is reused

    for i in range(len(X_synth)):
        # Pick neighbor with lowest usage count (promotes diversity)
        n1_idx = neighbor_idx[i, 0]
        n2_idx = neighbor_idx[i, 1]
        n1_usage = neighbor_usage.get(n1_idx, 0)
        n2_usage = neighbor_usage.get(n2_idx, 0)

        chosen_idx = n1_idx if n1_usage <= n2_usage else n2_idx
        neighbor_usage[chosen_idx] = neighbor_usage.get(chosen_idx, 0) + 1

        source_row = fraud_texts.iloc[chosen_idx]

        synth_text_rows.append({
            "product_id": str(start_id + i),
            "fraud_label": 1,
            "title_cleaned": source_row["title_cleaned"],
            "description_cleaned": source_row["description_cleaned"],
            "review1_cleaned": source_row["review1_cleaned"],
            "review2_cleaned": source_row["review2_cleaned"],
        })

    synth_text_df = pd.DataFrame(synth_text_rows)

    # Save
    out_path = PROCESSED_DIR / "synthetic_text_dataset.csv"
    synth_text_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(synth_text_df)} rows)")

    # Quality summary
    print(f"\n── Quality Summary ──")
    unique_titles = synth_text_df["title_cleaned"].nunique()
    print(f"Unique titles:     {unique_titles} / {len(synth_text_df)}")
    avg_title_len = synth_text_df["title_cleaned"].str.split().str.len().mean()
    avg_desc_len = synth_text_df["description_cleaned"].str.split().str.len().mean()
    print(f"Title avg length:  {avg_title_len:.1f} words")
    print(f"Desc  avg length:  {avg_desc_len:.1f} words")

    # Show reuse distribution
    reuse_counts = list(neighbor_usage.values())
    print(f"Real fraud reuse:  {len(reuse_counts)} unique sources")
    print(f"  min reuse: {min(reuse_counts)}, max reuse: {max(reuse_counts)}, "
          f"avg: {np.mean(reuse_counts):.1f}")

    return synth_text_df


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud data via SMOTE"
    )
    parser.add_argument(
        "--n-generate", type=int, default=150,
        help="Number of synthetic fraud rows to generate (default: 150)",
    )
    parser.add_argument(
        "--strategy", choices=["metadata", "text", "both"], default="both",
        help="Which modality to augment (default: both)",
    )
    args = parser.parse_args()

    if not SMOTE_AVAILABLE:
        print("ERROR: imbalanced-learn is required for SMOTE.")
        print("Install it with: pip install imbalanced-learn")
        return

    smote_meta_df = None

    if args.strategy in ("metadata", "both"):
        smote_meta_df = smote_metadata(n_generate=args.n_generate)

    if args.strategy in ("text", "both"):
        smote_text(n_generate=args.n_generate, smote_meta_df=smote_meta_df)

    print("\n" + "=" * 60)
    print("  DONE — Synthetic datasets regenerated")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Retrain metadata:  python models/train_metadata.py")
    print(f"  2. Retrain text:      Upload to Colab and run train_text_roberta_colab.ipynb")
    print(f"  3. Run ensemble:      python ensemble_evaluate.py")


if __name__ == "__main__":
    main()
