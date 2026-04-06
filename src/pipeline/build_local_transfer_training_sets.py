"""Build transfer-training modality datasets from labeled + pseudo-labeled local data.

This script uses modality-specific datasets only:
- local_labeled_text_dataset.csv
- local_labeled_image_dataset.csv
- local_labeled_metadata_dataset.csv
- local_unlabeled_text_dataset.csv
- local_unlabeled_image_dataset.csv
- local_unlabeled_metadata_dataset.csv

Pseudo labels are sourced from archived pseudo raw files in processed_data/local/pseudo_runs
(and optionally local_pseudo_labeled_raw.csv), then mapped by product_id onto unlabeled
modality datasets. No raw labeled dataset is used.

Outputs:
- processed_data/local/transfer_train/local_transfer_train_text_dataset.csv
- processed_data/local/transfer_train/local_transfer_train_image_dataset.csv
- processed_data/local/transfer_train/local_transfer_train_metadata_dataset.csv
- processed_data/local/transfer_train/local_transfer_train_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
LOCAL_DIR = (
    REPO_DIR / "processed_data" / "local"
    if (REPO_DIR / "processed_data" / "local").exists()
    else REPO_DIR / "data" / "local"
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "product_id" not in df.columns:
        raise ValueError(f"product_id column missing: {path}")
    df["product_id"] = df["product_id"].astype(str)
    return df


def parse_pseudo_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "product_id" not in df.columns:
        return pd.DataFrame(columns=["product_id", "fraud_label", "confidence", "source_file"])

    df["product_id"] = df["product_id"].astype(str)

    # Some pseudo_raw exports can contain duplicate fraud_label columns; pandas may
    # mangle them as fraud_label, fraud_label.1, fraud_label.2, ...
    # Keep the rightmost valid one.
    label_cols = [i for i, c in enumerate(df.columns) if str(c).startswith("fraud_label")]
    if not label_cols:
        return pd.DataFrame(columns=["product_id", "fraud_label", "confidence", "source_file"])

    label_series = None
    for idx in reversed(label_cols):
        cand = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        if cand.notna().any():
            label_series = cand
            break

    if label_series is None:
        return pd.DataFrame(columns=["product_id", "fraud_label", "confidence", "source_file"])

    out = pd.DataFrame({
        "product_id": df["product_id"],
        "fraud_label": label_series,
        "confidence": pd.to_numeric(df.get("confidence", 0.0), errors="coerce").fillna(0.0),
        "source_file": path.name,
    })
    out = out[out["fraud_label"].isin([0, 1])].copy()
    out["fraud_label"] = out["fraud_label"].astype(int)
    return out


def collect_pseudo_labels(local_dir: Path, pseudo_file: str, include_latest: bool) -> tuple[pd.DataFrame, list[Path]]:
    files: list[Path] = []

    if pseudo_file:
        files.append(Path(pseudo_file))
    else:
        runs_dir = local_dir / "pseudo_runs"
        if runs_dir.exists():
            auto_files = sorted(runs_dir.glob("*_pseudo_raw.csv"))
            auto_files = [p for p in auto_files if "TEST_" not in p.name.upper()]
            files.extend(auto_files)
        if include_latest:
            latest = local_dir / "local_pseudo_labeled_raw.csv"
            if latest.exists():
                files.append(latest)

    # Deduplicate file list while preserving order.
    seen = set()
    uniq_files = []
    for f in files:
        fp = str(f.resolve()) if f.exists() else str(f)
        if fp not in seen:
            seen.add(fp)
            uniq_files.append(f)

    if not uniq_files:
        raise FileNotFoundError("No pseudo label files found. Run pseudo-label generation first.")

    frames = [parse_pseudo_file(p) for p in uniq_files if p.exists()]
    pseudo = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["product_id", "fraud_label", "confidence", "source_file"]
    )

    if pseudo.empty:
        raise ValueError("Pseudo files were found but no valid pseudo labels (0/1) were parsed.")

    # Keep the highest-confidence pseudo label per product_id.
    pseudo = pseudo.sort_values(["confidence", "source_file"], ascending=[False, True])
    pseudo = pseudo.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)
    return pseudo, uniq_files


def apply_pseudo_to_unlabeled(unlabeled_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    merged = unlabeled_df.merge(
        pseudo_df[["product_id", "fraud_label", "confidence"]],
        on="product_id",
        how="inner",
        suffixes=("", "_pseudo"),
    )

    merged["fraud_label"] = merged["fraud_label_pseudo"].astype(int)
    merged = merged.drop(columns=[c for c in ["fraud_label_pseudo", "confidence"] if c in merged.columns])

    # Remove duplicated column names if present.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def combine_labeled_with_pseudo(labeled_df: pd.DataFrame, pseudo_modality_df: pd.DataFrame) -> pd.DataFrame:
    labeled_df = labeled_df.copy()
    labeled_df["fraud_label"] = pd.to_numeric(labeled_df["fraud_label"], errors="coerce")
    labeled_df = labeled_df[labeled_df["fraud_label"].isin([0, 1])].copy()
    labeled_df["fraud_label"] = labeled_df["fraud_label"].astype(int)

    labeled_ids = set(labeled_df["product_id"].astype(str))
    pseudo_only = pseudo_modality_df[~pseudo_modality_df["product_id"].astype(str).isin(labeled_ids)].copy()

    combined = pd.concat([labeled_df, pseudo_only], ignore_index=True)
    combined = combined.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local transfer-training modality datasets")
    parser.add_argument("--local-dir", type=str, default=str(LOCAL_DIR))
    parser.add_argument(
        "--pseudo-file",
        type=str,
        default="",
        help="Optional single pseudo_raw CSV path. If omitted, uses pseudo_runs/*.csv (+ latest pseudo file).",
    )
    parser.add_argument(
        "--include-latest-pseudo",
        action="store_true",
        help="Also include processed_data/local/local_pseudo_labeled_raw.csv when auto-collecting pseudo files.",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Keep holdout IDs in outputs (not recommended).",
    )
    parser.add_argument(
        "--exclude-holdout",
        action="store_true",
        default=False,
        help="Exclude IDs listed in final_holdout_ids.csv from transfer training outputs.",
    )
    parser.add_argument(
        "--holdout-ids",
        type=str,
        default="",
        help="Optional custom holdout IDs CSV path (defaults to processed_data/local/final_holdout/final_holdout_ids.csv).",
    )
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (local_dir / "transfer_train")
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_text = load_csv(local_dir / "local_labeled_text_dataset.csv")
    labeled_image = load_csv(local_dir / "local_labeled_image_dataset.csv")
    labeled_meta = load_csv(local_dir / "local_labeled_metadata_dataset.csv")

    unlabeled_text = load_csv(local_dir / "local_unlabeled_text_dataset.csv")
    unlabeled_image = load_csv(local_dir / "local_unlabeled_image_dataset.csv")
    unlabeled_meta = load_csv(local_dir / "local_unlabeled_metadata_dataset.csv")

    pseudo_df, pseudo_files = collect_pseudo_labels(
        local_dir=local_dir,
        pseudo_file=args.pseudo_file,
        include_latest=args.include_latest_pseudo,
    )

    pseudo_text = apply_pseudo_to_unlabeled(unlabeled_text, pseudo_df)
    pseudo_image = apply_pseudo_to_unlabeled(unlabeled_image, pseudo_df)
    pseudo_meta = apply_pseudo_to_unlabeled(unlabeled_meta, pseudo_df)

    train_text = combine_labeled_with_pseudo(labeled_text, pseudo_text)
    train_image = combine_labeled_with_pseudo(labeled_image, pseudo_image)
    train_meta = combine_labeled_with_pseudo(labeled_meta, pseudo_meta)

    # Optional holdout exclusion safety net.
    holdout_removed = 0
    holdout_path_used = ""
    exclude_holdout = args.exclude_holdout or (not args.include_holdout)
    if exclude_holdout:
        holdout_path = Path(args.holdout_ids) if args.holdout_ids else (local_dir / "final_holdout" / "final_holdout_ids.csv")
        if holdout_path.exists():
            holdout_df = pd.read_csv(holdout_path)
            if "product_id" in holdout_df.columns:
                holdout_ids = set(holdout_df["product_id"].astype(str))
                before = len(train_text)
                train_text = train_text[~train_text["product_id"].astype(str).isin(holdout_ids)].copy()
                train_image = train_image[~train_image["product_id"].astype(str).isin(holdout_ids)].copy()
                train_meta = train_meta[~train_meta["product_id"].astype(str).isin(holdout_ids)].copy()
                holdout_removed = before - len(train_text)
                holdout_path_used = str(holdout_path)

    # Keep only IDs present in all 3 modalities to preserve multimodal alignment.
    common_ids = (
        set(train_text["product_id"].astype(str))
        & set(train_image["product_id"].astype(str))
        & set(train_meta["product_id"].astype(str))
    )
    train_text = train_text[train_text["product_id"].astype(str).isin(common_ids)].sort_values("product_id").reset_index(drop=True)
    train_image = train_image[train_image["product_id"].astype(str).isin(common_ids)].sort_values("product_id").reset_index(drop=True)
    train_meta = train_meta[train_meta["product_id"].astype(str).isin(common_ids)].sort_values("product_id").reset_index(drop=True)

    text_out = out_dir / "local_transfer_train_text_dataset.csv"
    image_out = out_dir / "local_transfer_train_image_dataset.csv"
    meta_out = out_dir / "local_transfer_train_metadata_dataset.csv"
    summary_out = out_dir / "local_transfer_train_summary.json"

    train_text.to_csv(text_out, index=False)
    train_image.to_csv(image_out, index=False)
    train_meta.to_csv(meta_out, index=False)

    pseudo_counts = pseudo_df["fraud_label"].value_counts().to_dict()
    summary = {
        "inputs": {
            "local_dir": str(local_dir),
            "pseudo_files_used": [str(p) for p in pseudo_files],
            "holdout_ids": holdout_path_used,
        },
        "counts": {
            "pseudo_unique_ids": int(len(pseudo_df)),
            "pseudo_fraud": int(pseudo_counts.get(1, 0)),
            "pseudo_legit": int(pseudo_counts.get(0, 0)),
            "pseudo_text_rows": int(len(pseudo_text)),
            "pseudo_image_rows": int(len(pseudo_image)),
            "pseudo_metadata_rows": int(len(pseudo_meta)),
            "final_rows_aligned": int(len(train_text)),
            "holdout_rows_removed": int(holdout_removed),
            "final_fraud": int(train_text["fraud_label"].sum()) if len(train_text) else 0,
            "final_legit": int((train_text["fraud_label"] == 0).sum()) if len(train_text) else 0,
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

    print("Built transfer-training modality datasets:")
    print(f"  - {text_out}")
    print(f"  - {image_out}")
    print(f"  - {meta_out}")
    print(f"  - {summary_out}")
    print(f"Aligned rows: {len(train_text)}")


if __name__ == "__main__":
    main()
