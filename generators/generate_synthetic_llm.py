"""
Generate High-Quality Synthetic Fraud Listings via LLM

This script:
1. Reads real fraud examples from the original dataset
2. Builds a detailed prompt with few-shot examples
3. Outputs the prompt to a text file (for manual use with ChatGPT/Claude)
4. Provides a parser to convert LLM output back into properly formatted CSVs

Usage:
    # Step 1: Generate the prompt
    python generate_synthetic_llm.py --generate-prompt

    # Step 2: Copy the prompt from llm_prompt.txt → paste into ChatGPT/Claude
    #         Copy the LLM's response → save as llm_output.txt

    # Step 3: Parse LLM output into CSVs
    python generate_synthetic_llm.py --parse-output llm_output.txt

Output:
    processed_data/synthetic_text_dataset.csv    (text modality)
    processed_data/synthetic_metadata_dataset.csv (metadata modality)
"""

import re
import csv
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "Training_Data - Train.csv"
PROCESSED_DIR = BASE_DIR / "processed_data"

SEED = 42
NUM_TO_GENERATE = 150  # how many synthetic listings to request
NUM_EXAMPLES = 15       # how many real fraud examples to show as few-shot context

# ── Text preprocessing (mirrors preprocessing_pipeline.py) ──────

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
    NLTK_OK = True
except Exception:
    NLTK_OK = False
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
        "who", "when", "where", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "as",
    }
    LEMMATIZER = None


def clean_text(text):
    """Clean text the same way as preprocessing_pipeline.py."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if NLTK_OK and LEMMATIZER:
        try:
            tokens = word_tokenize(text)
            tokens = [
                LEMMATIZER.lemmatize(t) for t in tokens
                if t not in STOP_WORDS and len(t) > 1
            ]
            return " ".join(tokens)
        except Exception:
            pass

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def compute_derived_features(row):
    """Compute the same derived metadata features as preprocessing_pipeline.py."""
    listed = float(row.get("listed_price", 0))
    original = float(row.get("original_price", 0))
    seller_r = float(row.get("seller_rating", 0))
    item_r = float(row.get("item_rating", 0))
    r1 = float(row.get("review1_rating", 0))
    r2 = float(row.get("review2_rating", 0))

    price_dev = ((original - listed) / original * 100) if original > 0 else 0
    price_ratio = (listed / original) if original > 0 else 1
    abnormal = 1 if price_dev > 70 else 0
    review_diff = abs(r1 - r2)
    seller_gap = abs(seller_r - item_r)

    return {
        "price_deviation": round(price_dev, 6),
        "price_ratio": round(price_ratio, 6),
        "abnormal_discount": abnormal,
        "review_rating_diff": round(review_diff, 1),
        "seller_item_rating_gap": round(seller_gap, 1),
    }


# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Generate prompt
# ═══════════════════════════════════════════════════════════════════

def generate_prompt():
    """Build a detailed LLM prompt with real fraud examples."""
    print("Loading real fraud examples...")
    raw_df = pd.read_csv(RAW_DATA, dtype={"product_id": str})
    fraud_df = raw_df[raw_df["is_fraudulent"].astype(str).str.strip().str.upper() == "TRUE"]
    print(f"Found {len(fraud_df)} real fraud listings")

    # Sample diverse examples
    random.seed(SEED)
    examples = fraud_df.sample(n=min(NUM_EXAMPLES, len(fraud_df)), random_state=SEED)

    # Build few-shot examples
    example_blocks = []
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        block = f"""--- Example {i} ---
product_title: {row.get('product_title', '')}
description: {str(row.get('description', ''))[:500]}
listed_price: {row.get('listed_price', '')}
original_price: {row.get('original_price', '')}
seller_rating: {row.get('seller_rating', '')}
rating_count: {row.get('rating_count', '')}
item_rating: {row.get('item_rating', '')}
item_rating_count: {row.get('item_rating_count', '')}
review_1: {str(row.get('review_1', ''))[:300]}
review1_rating: {row.get('review1_rating', '')}
review_2: {str(row.get('review_2', ''))[:300]}
review2_rating: {row.get('review2_rating', '')}"""
        example_blocks.append(block)

    examples_text = "\n\n".join(example_blocks)

    # Get category distribution from real fraud
    titles = fraud_df["product_title"].dropna().tolist()
    # Extract rough categories
    categories = set()
    for t in titles:
        t_lower = str(t).lower()
        for kw in ["watch", "earbuds", "headphone", "speaker", "charger", "cable",
                    "tracker", "band", "lamp", "blender", "camera", "phone",
                    "tablet", "laptop", "keyboard", "mouse", "drone", "ring",
                    "necklace", "bracelet", "sunglasses", "bag", "wallet"]:
            if kw in t_lower:
                categories.add(kw)
    cat_list = ", ".join(sorted(categories)) if categories else "various electronics and accessories"

    prompt = f"""You are generating synthetic e-commerce product listings that represent FRAUDULENT listings for a fraud detection dataset. These listings should be realistic and diverse — they should look like real product pages, NOT obviously fake.

IMPORTANT GUIDELINES:
- Each listing must have a DETAILED product title (10-30 words, like a real Amazon/eBay title with specs)
- Each listing must have a LONG description (50-150 words, with bullet-point style features)
- Each listing must have TWO realistic reviews (20-60 words each, written like real customers)
- Reviews should be mixed — some positive, some negative, some mentioning quality issues
- Cover DIVERSE product categories: {cat_list}, plus clothing, home goods, beauty products, tools, toys, outdoor gear
- Vary the fraud patterns:
  * Some have inflated prices (listed >> original)
  * Some have suspiciously deep discounts (listed << original)
  * Some have few ratings but perfect scores
  * Some have mismatched seller/item ratings
  * Some have copied/generic descriptions
  * Some have reviews that don't match the product
- Make prices realistic (range: $5 - $500)
- Ratings should vary (seller: 1.0-5.0, item: 1.0-5.0, reviews: 1-5)
- Rating counts should vary (1 to 5000)

Here are {NUM_EXAMPLES} REAL fraud listings from the dataset for reference style:

{examples_text}

Now generate {NUM_TO_GENERATE} NEW synthetic fraud listings in the following JSON format.
Output ONLY a JSON array, no other text. Each object must have these fields:

[
  {{
    "product_title": "...",
    "description": "...",
    "listed_price": 99.99,
    "original_price": 49.99,
    "seller_rating": 4.5,
    "rating_count": 120,
    "item_rating": 3.8,
    "item_rating_count": 45,
    "review_1": "...",
    "review1_rating": 4,
    "review_2": "...",
    "review2_rating": 2
  }},
  ...
]

Generate exactly {NUM_TO_GENERATE} listings. Make them diverse and realistic."""

    # Save prompt
    prompt_path = BASE_DIR / "llm_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"\nPrompt saved to: {prompt_path}")
    print(f"Prompt length: {len(prompt):,} characters")
    print(f"\nInstructions:")
    print(f"  1. Open {prompt_path}")
    print(f"  2. Copy the entire contents")
    print(f"  3. Paste into ChatGPT (GPT-4), Claude, or Gemini")
    print(f"  4. Save the JSON response as: llm_output.txt")
    print(f"  5. Run: python generate_synthetic_llm.py --parse-output llm_output.txt")

    return prompt_path


# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Parse LLM output → CSVs
# ═══════════════════════════════════════════════════════════════════

def parse_llm_output(output_path):
    """Parse LLM JSON output into properly formatted synthetic CSVs."""
    print(f"Parsing LLM output from: {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Extract JSON array from response (handle markdown code blocks)
    json_match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
    else:
        # Try the whole file
        json_str = raw_text.strip()

    try:
        listings = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse JSON: {e}")
        print("Make sure the LLM output contains a valid JSON array.")
        print("Tip: Remove any text before [ and after ] in the file.")
        return

    print(f"Parsed {len(listings)} listings")

    # ── Build text dataset ──────────────────────────────────────
    text_rows = []
    meta_rows = []
    start_id = 30000  # offset from existing synthetic IDs (20000-20499)

    for i, item in enumerate(listings):
        pid = str(start_id + i)

        # Clean text fields (same pipeline as preprocessing_pipeline.py)
        title_clean = clean_text(item.get("product_title", ""))
        desc_clean = clean_text(item.get("description", ""))
        r1_clean = clean_text(item.get("review_1", ""))
        r2_clean = clean_text(item.get("review_2", ""))

        text_rows.append({
            "product_id": pid,
            "fraud_label": 1,
            "title_cleaned": title_clean,
            "description_cleaned": desc_clean,
            "review1_cleaned": r1_clean,
            "review2_cleaned": r2_clean,
        })

        # Build metadata row
        listed = float(item.get("listed_price", 0))
        original = float(item.get("original_price", 0))
        seller_r = float(item.get("seller_rating", 0))
        rating_cnt = int(item.get("rating_count", 0))
        item_r = float(item.get("item_rating", 0))
        item_r_cnt = int(item.get("item_rating_count", 0))
        r1_rating = float(item.get("review1_rating", 0))
        r2_rating = float(item.get("review2_rating", 0))

        derived = compute_derived_features(item)

        meta_row = {
            "product_id": pid,
            "fraud_label": 1,
            "listed_price": listed,
            "original_price": original,
            "seller_rating": seller_r,
            "rating_count": rating_cnt,
            "item_rating": item_r,
            "item_rating_count": item_r_cnt,
            "review1_rating": r1_rating,
            "review2_rating": r2_rating,
            **derived,
        }

        # Scaled versions (will be approximate — real scaling uses dataset min/max)
        # Load real metadata to get the scaling parameters
        meta_rows.append(meta_row)

    # ── Apply min-max scaling using real dataset stats ──────────
    real_meta = pd.read_csv(PROCESSED_DIR / "metadata_dataset.csv")
    features_to_scale = [
        "listed_price", "original_price", "price_deviation", "price_ratio",
        "seller_rating", "rating_count", "item_rating", "item_rating_count",
        "review1_rating", "review2_rating", "review_rating_diff", "seller_item_rating_gap",
    ]

    meta_df = pd.DataFrame(meta_rows)
    for col in features_to_scale:
        if col in real_meta.columns and col in meta_df.columns:
            min_val = real_meta[col].min()
            max_val = real_meta[col].max()
            if max_val > min_val:
                meta_df[f"{col}_scaled"] = (meta_df[col] - min_val) / (max_val - min_val)
            else:
                meta_df[f"{col}_scaled"] = 0

    # ── Save CSVs ───────────────────────────────────────────────
    text_df = pd.DataFrame(text_rows)
    text_path = PROCESSED_DIR / "synthetic_text_dataset.csv"
    meta_path = PROCESSED_DIR / "synthetic_metadata_dataset.csv"

    text_df.to_csv(text_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    print(f"\nSaved {len(text_df)} text rows to: {text_path}")
    print(f"Saved {len(meta_df)} metadata rows to: {meta_path}")

    # Quality check
    print(f"\n── Quality Summary ──")
    print(f"Title avg length: {text_df['title_cleaned'].str.split().str.len().mean():.1f} words")
    print(f"Desc  avg length: {text_df['description_cleaned'].str.split().str.len().mean():.1f} words")
    print(f"Review1 avg len:  {text_df['review1_cleaned'].str.split().str.len().mean():.1f} words")
    print(f"Review2 avg len:  {text_df['review2_cleaned'].str.split().str.len().mean():.1f} words")
    print(f"Price range:      ${meta_df['listed_price'].min():.2f} - ${meta_df['listed_price'].max():.2f}")
    print(f"Seller rating:    {meta_df['seller_rating'].min():.1f} - {meta_df['seller_rating'].max():.1f}")
    print(f"Unique titles:    {text_df['title_cleaned'].nunique()} / {len(text_df)}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    global NUM_TO_GENERATE

    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud listings via LLM"
    )
    parser.add_argument(
        "--generate-prompt", action="store_true",
        help="Generate the LLM prompt (Step 1)",
    )
    parser.add_argument(
        "--parse-output", type=str, default=None,
        help="Parse LLM JSON output file into CSVs (Step 2)",
    )
    parser.add_argument(
        "--num-generate", type=int, default=NUM_TO_GENERATE,
        help=f"Number of listings to request (default: {NUM_TO_GENERATE})",
    )
    args = parser.parse_args()

    NUM_TO_GENERATE = args.num_generate

    if args.generate_prompt:
        generate_prompt()
    elif args.parse_output:
        parse_llm_output(args.parse_output)
    else:
        parser.print_help()
        print("\n  Step 1: python generate_synthetic_llm.py --generate-prompt")
        print("  Step 2: python generate_synthetic_llm.py --parse-output llm_output.txt")


if __name__ == "__main__":
    main()
