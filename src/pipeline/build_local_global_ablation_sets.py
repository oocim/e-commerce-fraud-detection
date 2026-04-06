"""Build ablation transfer datasets by mixing former(global) and local transfer data.

Default mix ratio is former:local = 40:60.
Input local datasets are the pure local transfer sets produced by
build_local_transfer_training_sets.py.

Outputs are written to a separate directory (default):
  processed_data/local/transfer_train_ablation_40_60/
with 3 modality files:
  - local_transfer_train_ablation_text_dataset.csv
  - local_transfer_train_ablation_image_dataset.csv
  - local_transfer_train_ablation_metadata_dataset.csv
and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = (
    REPO_DIR / "data" / "processed"
    if (REPO_DIR / "data" / "processed").exists()
    else REPO_DIR / "processed_data"
)
LOCAL_DIR = (
    REPO_DIR / "processed_data" / "local"
    if (REPO_DIR / "processed_data" / "local").exists()
    else REPO_DIR / "data" / "local"
)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "product_id" not in df.columns:
        raise ValueError(f"Missing product_id column: {path}")
    if "fraud_label" not in df.columns:
        raise ValueError(f"Missing fraud_label column: {path}")
    df["product_id"] = df["product_id"].astype(str)
    df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
    df = df[df["fraud_label"].isin([0, 1])].copy()
    df["fraud_label"] = df["fraud_label"].astype(int)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy()

    class_counts = df["fraud_label"].value_counts().to_dict()
    n_fraud_src = int(class_counts.get(1, 0))
    n_legit_src = int(class_counts.get(0, 0))
    if n_fraud_src == 0 or n_legit_src == 0:
        return df.sample(n=n, random_state=seed)

    frac_fraud = n_fraud_src / len(df)
    target_fraud = int(round(n * frac_fraud))
    target_fraud = min(target_fraud, n_fraud_src)
    target_legit = n - target_fraud
    if target_legit > n_legit_src:
        target_legit = n_legit_src
        target_fraud = n - target_legit
    if target_fraud > n_fraud_src:
        target_fraud = n_fraud_src
        target_legit = n - target_fraud

    fraud_df = df[df["fraud_label"] == 1].sample(n=target_fraud, random_state=seed)
    legit_df = df[df["fraud_label"] == 0].sample(n=target_legit, random_state=seed)
    out = pd.concat([fraud_df, legit_df], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_modality_mix(
    local_df: pd.DataFrame,
    global_df: pd.DataFrame,
    former_ratio: float,
    local_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    local_n = len(local_df)
    if local_n == 0:
        raise ValueError("Local transfer dataset is empty; cannot build ablation mix")

    # former_n / local_n = former_ratio / local_ratio
    target_former_n = int(round(local_n * (former_ratio / local_ratio)))

    # Avoid accidental overlap by product_id.
    local_ids = set(local_df["product_id"].astype(str))
    global_pool = global_df[~global_df["product_id"].astype(str).isin(local_ids)].copy()

    former_sample = stratified_sample(global_pool, target_former_n, seed)

    mixed = pd.concat([local_df, former_sample], ignore_index=True)
    mixed = mixed.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)

    stats = {
        "local_rows": int(local_n),
        "former_target_rows": int(target_former_n),
        "former_sampled_rows": int(len(former_sample)),
        "mixed_rows": int(len(mixed)),
        "mixed_fraud": int((mixed["fraud_label"] == 1).sum()),
        "mixed_legit": int((mixed["fraud_label"] == 0).sum()),
        "local_fraction_in_mixed": float(local_n / len(mixed)) if len(mixed) else 0.0,
        "former_fraction_in_mixed": float(len(former_sample) / len(mixed)) if len(mixed) else 0.0,
    }
    return mixed, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation datasets with former/global + local transfer mix")
    parser.add_argument("--former-ratio", type=float, default=0.4, help="Former(global) ratio component")
    parser.add_argument("--local-ratio", type=float, default=0.6, help="Local ratio component")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-transfer-dir",
        type=str,
        default=str(LOCAL_DIR / "transfer_train"),
        help="Directory containing pure local transfer modality datasets",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(LOCAL_DIR / "transfer_train_ablation_40_60"),
        help="Output directory for ablation modality datasets",
    )
    args = parser.parse_args()

    if args.former_ratio <= 0 or args.local_ratio <= 0:
        raise ValueError("former-ratio and local-ratio must be > 0")

    local_dir = Path(args.local_transfer_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_text = load_df(local_dir / "local_transfer_train_text_dataset.csv")
    local_image = load_df(local_dir / "local_transfer_train_image_dataset.csv")
    local_meta = load_df(local_dir / "local_transfer_train_metadata_dataset.csv")

    global_text = load_df(PROCESSED_DIR / "text_dataset.csv")
    global_image = load_df(PROCESSED_DIR / "image_dataset.csv")
    global_meta = load_df(PROCESSED_DIR / "metadata_dataset.csv")

    mixed_text, stats_text = build_modality_mix(
        local_text, global_text, args.former_ratio, args.local_ratio, args.seed
    )
    mixed_image, stats_image = build_modality_mix(
        local_image, global_image, args.former_ratio, args.local_ratio, args.seed
    )
    mixed_meta, stats_meta = build_modality_mix(
        local_meta, global_meta, args.former_ratio, args.local_ratio, args.seed
    )

    text_out = out_dir / "local_transfer_train_ablation_text_dataset.csv"
    image_out = out_dir / "local_transfer_train_ablation_image_dataset.csv"
    meta_out = out_dir / "local_transfer_train_ablation_metadata_dataset.csv"
    summary_out = out_dir / "local_transfer_train_ablation_summary.json"

    mixed_text.to_csv(text_out, index=False)
    mixed_image.to_csv(image_out, index=False)
    mixed_meta.to_csv(meta_out, index=False)

    summary = {
        "config": {
            "former_ratio": args.former_ratio,
            "local_ratio": args.local_ratio,
            "seed": args.seed,
            "local_transfer_dir": str(local_dir),
        },
        "modalities": {
            "text": stats_text,
            "image": stats_image,
            "metadata": stats_meta,
        },
        "outputs": {
            "text": str(text_out),
            "image": str(image_out),
            "metadata": str(meta_out),
            "summary": str(summary_out),
        },
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Built ablation modality datasets:")
    print(f"  - {text_out}")
    print(f"  - {image_out}")
    print(f"  - {meta_out}")
    print(f"  - {summary_out}")


if __name__ == "__main__":
    main()
