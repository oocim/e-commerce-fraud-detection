"""
Multimodal Fraud Detection — Inference Pipeline

Takes raw product data (same format as Training_Data - Train.csv) and classifies
each listing as fraudulent or not by combining predictions from three modalities:
  1. Text   — Hybrid RoBERTa + TF-IDF
  2. Image  — ResNet-50 (CNN)
  3. Metadata — XGBoost / Random Forest / Logistic Regression

Usage:
    # Classify a CSV file
    python predict.py --input new_products.csv --output results.csv

    # Classify a single product interactively
    python predict.py --interactive

Required saved artifacts (in saved_models/):
    text/hybrid_roberta_tfidf.pth     — RoBERTa+TF-IDF model weights
    text/tokenizer/                   — RoBERTa tokenizer files
    text/tfidf_vectorizer.joblib      — Fitted TF-IDF vectorizer
    image/best_resnet50.pth           — ResNet-50 model weights
    metadata/xgboost.joblib           — Best metadata model (or RF/LR)
    metadata/scaler.joblib            — StandardScaler for metadata
    ensemble_weights.json             — (auto-generated) modality weights
"""

import io
import re
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models

try:
    from transformers import RobertaTokenizer, RobertaModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from PIL import Image
    import requests
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "saved_models"


# ═══════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING (mirrors preprocessing_pipeline.py)
# ═══════════════════════════════════════════════════════════════════

class TextPreprocessor:
    """Cleans text the same way as the training preprocessing pipeline."""

    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)
                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None

    def clean(self, text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                tokens = word_tokenize(text)
                tokens = [
                    self.lemmatizer.lemmatize(t)
                    for t in tokens
                    if t not in self.stop_words and len(t) > 1
                ]
                return " ".join(tokens)
            except Exception:
                pass

        # Fallback basic cleaning
        basic_sw = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "what", "which", "who", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "as",
        }
        tokens = text.split()
        tokens = [t for t in tokens if t not in basic_sw and len(t) > 1]
        return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (must match training architectures)
# ═══════════════════════════════════════════════════════════════════

class RoBERTaTfidfFraudModel(nn.Module):
    """Hybrid RoBERTa [CLS] + TF-IDF → classifier (mirrors training notebook)."""

    def __init__(self, model_name="roberta-base", tfidf_dim=2000, num_labels=2, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        roberta_dim = self.roberta.config.hidden_size  # 768

        self.tfidf_proj = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fused_dim = roberta_dim + 256

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, tfidf_features, labels=None):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = roberta_out.last_hidden_state[:, 0, :]
        tfidf_emb = self.tfidf_proj(tfidf_features)
        fused = torch.cat([cls_emb, tfidf_emb], dim=-1)
        logits = self.classifier(fused)
        return {"logits": logits}


def build_resnet50(num_classes=2):
    """Recreate the ResNet-50 architecture from training."""
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ═══════════════════════════════════════════════════════════════════
#  FRAUD DETECTOR (loads all models and runs inference)
# ═══════════════════════════════════════════════════════════════════

class FraudDetector:
    """
    Multimodal fraud detection system.

    Loads trained models for text, image, and metadata modalities,
    preprocesses raw product data, runs inference, and combines
    predictions with a weighted ensemble.
    """

    def __init__(self, models_dir=None, device=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_preprocessor = TextPreprocessor()

        # Placeholders
        self.text_model = None
        self.tokenizer = None
        self.tfidf = None
        self.image_model = None
        self.metadata_model = None
        self.metadata_model_type = ""  # "xgboost", "random_forest", or "logistic_regression"
        self.metadata_scaler = None
        self.metadata_minmax = None  # min/max stats for _scaled features
        self.weights = {"text": 0.475, "image": 0.05, "metadata": 0.475}

        # Temperature scaling for probability calibration (>1 = softer, <1 = sharper)
        # text=1.0: model already calibrated via label smoothing + layer freezing
        # image=1.0: normal
        # metadata=1.2: slight softening for XGBoost sharp outputs
        self.temperature = {"text": 1.0, "image": 1.0, "metadata": 1.2}

        self._loaded = {"text": False, "image": False, "metadata": False}

    def load_models(self):
        """Load all available models. Skips modalities with missing files."""
        print(f"Loading models from {self.models_dir} ...")
        print(f"Device: {self.device}\n")

        self._load_text_model()
        self._load_image_model()
        self._load_metadata_model()
        self._load_weights()

        loaded = [k for k, v in self._loaded.items() if v]
        missing = [k for k, v in self._loaded.items() if not v]
        print(f"\nLoaded modalities: {loaded}")
        if missing:
            print(f"Missing modalities (will be skipped): {missing}")
        if not loaded:
            raise RuntimeError("No models loaded. Cannot make predictions.")

    def _load_text_model(self):
        text_dir = self.models_dir / "text"
        weights_path = text_dir / "hybrid_roberta_tfidf.pth"
        tfidf_path = text_dir / "tfidf_vectorizer.joblib"
        tokenizer_dir = text_dir / "tokenizer"

        if not (weights_path.exists() and tfidf_path.exists() and HF_AVAILABLE):
            print("[Text] Model files not found or transformers not installed — skipping")
            return

        self.tfidf = joblib.load(tfidf_path)
        tfidf_dim = len(self.tfidf.vocabulary_)

        # Try loading tokenizer from subfolder, fallback to text_dir itself
        tok_path = str(tokenizer_dir) if tokenizer_dir.exists() else str(text_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(tok_path)

        self.text_model = RoBERTaTfidfFraudModel(
            model_name="roberta-base", tfidf_dim=tfidf_dim
        )
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.text_model.load_state_dict(state)
        self.text_model.to(self.device).eval()
        self._loaded["text"] = True
        print(f"[Text] Loaded RoBERTa+TF-IDF (TF-IDF dim={tfidf_dim})")

    def _load_image_model(self):
        img_path = self.models_dir / "image" / "best_resnet50.pth"
        if not (img_path.exists() and PIL_AVAILABLE):
            print("[Image] Model file not found or Pillow not installed — skipping")
            return

        self.image_model = build_resnet50()
        state = torch.load(img_path, map_location=self.device, weights_only=False)
        self.image_model.load_state_dict(state)
        self.image_model.to(self.device).eval()
        self._loaded["image"] = True
        print("[Image] Loaded ResNet-50")

    def _load_metadata_model(self):
        meta_dir = self.models_dir / "metadata"

        # Try loading in preference order: xgboost > random_forest > logistic_regression
        model_path = None
        for name in ["xgboost.joblib", "random_forest.joblib", "logistic_regression.joblib"]:
            candidate = meta_dir / name
            if candidate.exists():
                model_path = candidate
                break

        scaler_path = meta_dir / "scaler.joblib"
        if model_path is None:
            print("[Metadata] No model .joblib found — skipping")
            return

        self.metadata_model = joblib.load(model_path)
        self.metadata_model_type = model_path.stem  # e.g. "xgboost", "logistic_regression"
        if scaler_path.exists():
            self.metadata_scaler = joblib.load(scaler_path)
        # Load min/max stats for _scaled features (saved during training)
        minmax_path = meta_dir / "minmax_stats.json"
        if minmax_path.exists():
            with open(minmax_path) as f:
                self.metadata_minmax = json.load(f)
            print(f"[Metadata] Loaded min/max scaling stats")
        self._loaded["metadata"] = True
        print(f"[Metadata] Loaded {model_path.stem}")

    def _load_weights(self):
        w_path = self.models_dir / "ensemble_weights.json"
        if w_path.exists():
            with open(w_path) as f:
                config = json.load(f)
            if "weights" in config:
                self.weights = config["weights"]
            elif all(k in config for k in ["text", "image", "metadata"]):
                self.weights = config
            if "temperature" in config:
                self.temperature = config["temperature"]
            print(f"[Ensemble] Loaded weights: {self.weights}")
            print(f"[Ensemble] Loaded temperature: {self.temperature}")
        else:
            # Default: text & metadata get higher weight, image lower
            default_w = {"text": 0.475, "image": 0.05, "metadata": 0.475}
            for k in self._loaded:
                self.weights[k] = default_w[k] if self._loaded[k] else 0.0
            print(f"[Ensemble] Using default weights: {self.weights}")
            print(f"[Ensemble] Temperature scaling: {self.temperature}")

    # ── Preprocessing ────────────────────────────────────────────

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw product DataFrame (same columns as Training_Data).

        Expected columns: product_id, product_title, description,
            review_1, review1_rating, review_2, review2_rating,
            listed_price, original_price, seller_rating, rating_count,
            item_rating, item_rating_count, image_url
        """
        out = df.copy()

        # Ensure product_id
        if "product_id" not in out.columns:
            out["product_id"] = [f"new_{i+1:05d}" for i in range(len(out))]

        # Clean text fields
        text_map = {
            "product_title": "title_cleaned",
            "description": "description_cleaned",
            "review_1": "review1_cleaned",
            "review_2": "review2_cleaned",
        }
        for raw, clean in text_map.items():
            if raw in out.columns:
                out[clean] = out[raw].apply(self.text_preprocessor.clean)
            else:
                out[clean] = ""

        # Combined text for RoBERTa input
        out["text"] = (
            "Title: " + out["title_cleaned"]
            + " Description: " + out["description_cleaned"]
            + " Review1: " + out["review1_cleaned"]
            + " Review2: " + out["review2_cleaned"]
        )

        # Parse numeric metadata
        num_cols = [
            "listed_price", "original_price", "seller_rating", "rating_count",
            "item_rating", "item_rating_count", "review1_rating", "review2_rating",
        ]
        for col in num_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(
                    out[col].astype(str).str.replace(",", "").str.replace("$", ""),
                    errors="coerce",
                )

        # Derived metadata features (same as preprocessing_pipeline.py)
        lp = out.get("listed_price", pd.Series(dtype=float))
        op = out.get("original_price", pd.Series(dtype=float))
        out["price_deviation"] = np.where(op > 0, ((op - lp) / op) * 100, 0.0)
        out["price_ratio"] = np.where(op > 0, lp / op, 1.0)
        out["abnormal_discount"] = (out["price_deviation"] > 70).astype(int)
        r1 = out.get("review1_rating", pd.Series(0.0, index=out.index))
        r2 = out.get("review2_rating", pd.Series(0.0, index=out.index))
        out["review_rating_diff"] = (r1 - r2).abs()
        sr = out.get("seller_rating", pd.Series(0.0, index=out.index))
        ir = out.get("item_rating", pd.Series(0.0, index=out.index))
        out["seller_item_rating_gap"] = (sr - ir).abs()

        # Min-max scaled features (use training-range stats if available, else approximate)
        scale_cols = [
            "listed_price", "original_price", "price_deviation", "price_ratio",
            "seller_rating", "rating_count", "item_rating", "item_rating_count",
            "review1_rating", "review2_rating", "review_rating_diff",
            "seller_item_rating_gap",
        ]
        for col in scale_cols:
            if col in out.columns:
                if self.metadata_minmax and col in self.metadata_minmax:
                    # Use training dataset's min/max (correct for any batch size)
                    cmin = self.metadata_minmax[col]["min"]
                    cmax = self.metadata_minmax[col]["max"]
                else:
                    # Fallback: use batch stats (only works for large batches)
                    cmin, cmax = out[col].min(), out[col].max()
                if cmax > cmin:
                    out[f"{col}_scaled"] = (out[col] - cmin) / (cmax - cmin)
                else:
                    out[f"{col}_scaled"] = 0.5  # neutral value instead of 0

        return out

    # ── Per-modality predictions ─────────────────────────────────

    def predict_text(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities from the text model."""
        if not self._loaded["text"]:
            return np.full(len(df), np.nan)

        texts = df["text"].fillna("").tolist()

        # TF-IDF
        tfidf_feats = self.tfidf.transform(texts).toarray().astype(np.float32)
        tfidf_tensor = torch.tensor(tfidf_feats, dtype=torch.float32)

        # Tokenize
        encodings = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt",
        )

        probas = []
        batch_size = 8
        n = len(texts)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                batch = {
                    "input_ids": encodings["input_ids"][i:j].to(self.device),
                    "attention_mask": encodings["attention_mask"][i:j].to(self.device),
                    "tfidf_features": tfidf_tensor[i:j].to(self.device),
                }
                out = self.text_model(**batch)
                logits = out["logits"] / self.temperature["text"]
                probs = torch.softmax(logits, dim=-1)[:, 1]
                probas.extend(probs.cpu().numpy())

        return np.array(probas)

    def predict_image(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities from the image model."""
        if not self._loaded["image"]:
            return np.full(len(df), np.nan)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        headers = {"User-Agent": "Mozilla/5.0"}
        probas = []

        for _, row in df.iterrows():
            url = row.get("image_url", "")
            try:
                resp = requests.get(str(url), headers=headers, timeout=15)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.image_model(img_tensor)
                    logits = out / self.temperature["image"]
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                probas.append(prob)
            except Exception:
                probas.append(np.nan)

        return np.array(probas)

    def predict_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities from the metadata model."""
        if not self._loaded["metadata"]:
            return np.full(len(df), np.nan)

        # Use the same feature columns as training
        raw_features = [
            "listed_price", "original_price", "seller_rating", "rating_count",
            "item_rating", "item_rating_count", "review1_rating", "review2_rating",
            "price_deviation", "price_ratio", "abnormal_discount",
            "review_rating_diff", "seller_item_rating_gap",
        ]
        scaled_features = [f"{c}_scaled" for c in [
            "listed_price", "original_price", "price_deviation", "price_ratio",
            "seller_rating", "rating_count", "item_rating", "item_rating_count",
            "review1_rating", "review2_rating", "review_rating_diff",
            "seller_item_rating_gap",
        ]]
        feature_cols = raw_features + scaled_features

        # Use only columns that exist
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values

        # Only apply StandardScaler for logistic_regression.
        # XGBoost and Random Forest were trained on unscaled features.
        if self.metadata_scaler is not None and getattr(self, 'metadata_model_type', '') == 'logistic_regression':
            try:
                X = self.metadata_scaler.transform(X)
            except ValueError:
                pass  # Shape mismatch — use unscaled

        raw_proba = self.metadata_model.predict_proba(X)[:, 1]
        # Apply temperature scaling to metadata probabilities
        # Convert proba → logit → scale → back to proba
        t = self.temperature["metadata"]
        if t != 1.0:
            eps = 1e-7
            clipped = np.clip(raw_proba, eps, 1 - eps)
            logits = np.log(clipped / (1 - clipped))  # inverse sigmoid
            scaled = logits / t
            raw_proba = 1 / (1 + np.exp(-scaled))  # sigmoid
        return raw_proba

    # ── Ensemble ─────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
        """
        Run full multimodal inference on a raw product DataFrame.

        Returns DataFrame with:
          product_id, text_proba, image_proba, metadata_proba,
          ensemble_proba, fraud_prediction, confidence
        """
        processed = self.preprocess_raw(df)

        print(f"\nRunning inference on {len(processed)} products ...")

        # Per-modality predictions
        p_text = self.predict_text(processed)
        p_image = self.predict_image(processed)
        p_meta = self.predict_metadata(processed)

        # Weighted average (skip NaN modalities per row)
        probas = np.column_stack([p_text, p_image, p_meta])
        wts = np.array([self.weights["text"], self.weights["image"], self.weights["metadata"]])

        ensemble_proba = np.zeros(len(processed))
        for i in range(len(processed)):
            valid = ~np.isnan(probas[i])
            if valid.any():
                w = wts[valid]
                p = probas[i][valid]
                ensemble_proba[i] = np.dot(w, p) / w.sum()
            else:
                ensemble_proba[i] = 0.5  # No model available — uncertain

        fraud_pred = (ensemble_proba >= threshold).astype(int)

        result = pd.DataFrame({
            "product_id": processed["product_id"].values,
            "text_fraud_proba": np.round(p_text, 4),
            "image_fraud_proba": np.round(p_image, 4),
            "metadata_fraud_proba": np.round(p_meta, 4),
            "ensemble_fraud_proba": np.round(ensemble_proba, 4),
            "fraud_prediction": fraud_pred,
            "confidence": np.round(np.abs(ensemble_proba - 0.5) * 2, 4),
        })

        # Add ground truth if available
        if "is_fraudulent" in df.columns:
            label_map = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
            result["actual_label"] = (
                df["is_fraudulent"].astype(str).str.strip().str.upper().map(label_map)
            )
        if "fraud_label" in df.columns:
            result["actual_label"] = df["fraud_label"]

        return result

    def print_results(self, result: pd.DataFrame):
        """Pretty-print classification results."""
        print("\n" + "=" * 72)
        print("  FRAUD DETECTION RESULTS")
        print("=" * 72)

        for _, row in result.iterrows():
            pid = row["product_id"]
            pred = "FRAUD" if row["fraud_prediction"] == 1 else "LEGITIMATE"
            conf = row["confidence"] * 100
            emoji = "!!" if pred == "FRAUD" else "OK"

            print(f"\n[{emoji}] Product {pid}: {pred} (confidence: {conf:.1f}%)")
            print(f"    Text model:     {row['text_fraud_proba']:.4f}")
            print(f"    Image model:    {row['image_fraud_proba']:.4f}")
            print(f"    Metadata model: {row['metadata_fraud_proba']:.4f}")
            print(f"    Ensemble:       {row['ensemble_fraud_proba']:.4f}")

            if "actual_label" in row and pd.notna(row["actual_label"]):
                actual = "FRAUD" if row["actual_label"] == 1 else "LEGITIMATE"
                match = "CORRECT" if (row["fraud_prediction"] == row["actual_label"]) else "WRONG"
                print(f"    Actual:         {actual} ({match})")

        n_fraud = (result["fraud_prediction"] == 1).sum()
        print(f"\n{'=' * 72}")
        print(f"Total: {len(result)} products | {n_fraud} flagged as fraud | "
              f"{len(result) - n_fraud} legitimate")
        print(f"{'=' * 72}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def interactive_mode(detector):
    """Classify a single product from user input."""
    print("\n--- Enter product details (press Enter to skip optional fields) ---\n")

    data = {}
    data["product_id"] = input("Product ID [auto]: ").strip() or "interactive_001"
    data["product_title"] = input("Product title: ").strip()
    data["description"] = input("Description: ").strip()
    data["listed_price"] = input("Listed price: ").strip()
    data["original_price"] = input("Original price: ").strip()
    data["seller_rating"] = input("Seller rating (0-5): ").strip()
    data["rating_count"] = input("Rating count: ").strip()
    data["item_rating"] = input("Item rating (0-5): ").strip()
    data["item_rating_count"] = input("Item rating count: ").strip()
    data["review_1"] = input("Review 1 text: ").strip()
    data["review1_rating"] = input("Review 1 rating (1-5): ").strip()
    data["review_2"] = input("Review 2 text: ").strip()
    data["review2_rating"] = input("Review 2 rating (1-5): ").strip()
    data["image_url"] = input("Image URL: ").strip()

    df = pd.DataFrame([data])
    result = detector.predict(df)
    detector.print_results(result)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Fraud Detection — Classify product listings"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to CSV file with raw product data (same format as training data)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="fraud_predictions.csv",
        help="Output path for predictions CSV",
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Directory containing saved models (default: saved_models/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.60,
        help="Fraud probability threshold (default: 0.60)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Classify a single product interactively",
    )
    args = parser.parse_args()

    detector = FraudDetector(models_dir=args.models_dir)
    detector.load_models()

    if args.interactive:
        interactive_mode(detector)
        return

    if args.input is None:
        parser.print_help()
        print("\nProvide --input <file.csv> or --interactive")
        return

    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    df = pd.read_csv(input_path, dtype={"product_id": str})
    print(f"Loaded {len(df)} products from {input_path}")

    # Run inference
    result = detector.predict(df, threshold=args.threshold)

    # Display results
    detector.print_results(result)

    # Save
    output_path = Path(args.output)
    result.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # If ground truth available, print summary metrics
    if "actual_label" in result.columns and result["actual_label"].notna().all():
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        y_true = result["actual_label"].astype(int)
        y_pred = result["fraud_prediction"]
        print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print("\n" + classification_report(y_true, y_pred,
              target_names=["Not Fraud", "Fraud"]))


if __name__ == "__main__":
    main()
