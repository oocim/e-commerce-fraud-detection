"""
Generate 150 diverse synthetic FRAUD listings using template-based generation.

Unlike SMOTE (which copies nearest-neighbor text verbatim), this creates
genuinely new fraudulent-looking text by combining randomized components
that mirror real-world e-commerce fraud patterns:
  - Keyword-stuffed titles with urgency/superlative spam
  - Vague/generic descriptions lacking real specs
  - Bot-like 5-star reviews and contradicting negative reviews
  - Mismatched prices, ratings, and review patterns

Output:
  processed_data/synthetic_text_dataset.csv
  processed_data/synthetic_metadata_dataset.csv
"""

import re
import random
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
SEED = 42
NUM_GENERATE = 150
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"

random.seed(SEED)
np.random.seed(SEED)

# ── NLP setup (mirrors preprocessing_pipeline.py) ──────────────
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
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
            return " ".join(tokens)
        except Exception:
            pass
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ═════════════════════════════════════════════════════════════════
#  COMPONENT POOLS — randomized building blocks for fraud patterns
# ═════════════════════════════════════════════════════════════════

PRODUCT_CATEGORIES = [
    # (category, base product names, typical legit price range)
    ("smartwatch", [
        "Smart Watch Fitness Tracker with Heart Rate Monitor",
        "Bluetooth Smartwatch with Blood Pressure SpO2 Sleep Monitor",
        "Sport Smart Watch with Step Counter Calorie Tracker",
        "Digital Smart Watch with Touch Screen AMOLED Display",
        "Fitness Smartwatch Activity Tracker Water Resistant IP68",
        "Health Monitor Smart Watch with ECG Blood Oxygen",
    ], (15, 80)),
    ("earbuds", [
        "Wireless Earbuds Bluetooth 5.3 with Charging Case",
        "True Wireless Noise Cancelling Earbuds Hi-Fi Stereo",
        "Bluetooth Earbuds with Microphone Deep Bass Audio",
        "Sport Wireless Earphones Waterproof IPX7 Running",
        "Premium Active Noise Cancelling Wireless Earbuds",
        "TWS Bluetooth Earbuds with LED Display Charging Case",
    ], (10, 60)),
    ("headphones", [
        "Over Ear Wireless Headphones with Active Noise Cancellation",
        "Bluetooth Headphones Foldable with Built-in Microphone",
        "Gaming Headset with 7.1 Surround Sound RGB Lights",
        "Professional Studio Monitor Headphones Wired",
        "Noise Cancelling Wireless Headphones 60H Battery Life",
    ], (20, 80)),
    ("speaker", [
        "Portable Bluetooth Speaker Waterproof IPX7 Outdoor",
        "Wireless Mini Speaker with LED Lights Party Mode",
        "Bluetooth Soundbar for TV with Subwoofer",
        "Smart Speaker with Voice Assistant Built-in WiFi",
        "Portable Party Speaker with Bass Boost RGB Lights",
    ], (15, 70)),
    ("charger", [
        "Fast Wireless Charging Pad 15W Compatible with All Phones",
        "USB C Fast Charger 65W GaN with Multiple Ports",
        "Portable Power Bank 20000mAh with Quick Charge 3.0",
        "Solar Power Bank 30000mAh Waterproof with LED Flashlight",
        "Magnetic Wireless Charger Stand for Phone and Watch",
    ], (8, 45)),
    ("camera", [
        "4K Action Camera Waterproof with WiFi Remote Control",
        "Mini Security Camera HD 1080P Indoor WiFi Night Vision",
        "Trail Camera 32MP HD Wildlife Hunting Game Camera",
        "Dash Cam Front and Rear Dual Camera with Night Vision",
        "Ring Camera Doorbell with Two Way Audio Motion Detection",
    ], (20, 90)),
    ("phone_accessory", [
        "Phone Case Shockproof Military Grade Drop Protection",
        "Screen Protector Tempered Glass 9H Hardness Anti Scratch",
        "Car Phone Mount Magnetic Universal Dashboard Holder",
        "Phone Gimbal Stabilizer 3 Axis for Video Recording",
        "Selfie Stick Tripod with Bluetooth Remote Extendable",
    ], (5, 35)),
    ("laptop_accessory", [
        "Laptop Stand Adjustable Aluminum Ergonomic Riser",
        "USB C Hub Docking Station 12 in 1 with HDMI Ethernet",
        "Mechanical Gaming Keyboard RGB Backlit with Blue Switches",
        "Wireless Mouse Ergonomic Vertical with USB Receiver",
        "Laptop Cooling Pad with 5 Fans Gaming Notebook Cooler",
    ], (12, 55)),
    ("home", [
        "LED Desk Lamp with Wireless Charging USB Port Dimmable",
        "Air Purifier HEPA Filter for Large Room with Timer",
        "Robot Vacuum Cleaner with Mop Self Charging Smart Mapping",
        "Electric Space Heater with Thermostat Energy Efficient",
        "Smart LED Light Bulbs WiFi Color Changing RGBW 4 Pack",
    ], (15, 85)),
    ("beauty", [
        "Facial Cleansing Brush Electric Waterproof Deep Pore",
        "LED Face Mask Light Therapy 7 Color Anti Aging",
        "Hair Straightener Flat Iron with Ceramic Plates",
        "Electric Scalp Massager Waterproof with 4 Heads",
        "Teeth Whitening Kit with LED Light Professional Grade",
    ], (10, 60)),
    ("outdoor", [
        "Tactical Flashlight LED Rechargeable Super Bright 10000 Lumens",
        "Survival Kit Emergency 250 Pieces First Aid Camping",
        "Binoculars 12x42 HD with Phone Adapter Night Vision",
        "Camping Hammock Double Portable with Tree Straps",
        "Hiking Backpack 50L Waterproof with Rain Cover",
    ], (15, 70)),
    ("kitchen", [
        "Portable Blender Personal Size USB Rechargeable",
        "Air Fryer 5.5 Quart Digital with 8 Presets",
        "Electric Kettle with Temperature Control Stainless Steel",
        "Vacuum Sealer Machine with Starter Kit Bags Included",
        "Coffee Scale with Timer 0.1g High Precision",
    ], (12, 65)),
    ("toys", [
        "RC Drone with 4K Camera FPV GPS Foldable Quadcopter",
        "Building Blocks Set 1000 Pieces STEM Educational Toy",
        "Remote Control Car Off Road Monster Truck 4WD",
        "Robot Dog Toy Interactive with Voice Commands",
        "Electric Water Gun Automatic High Capacity",
    ], (15, 80)),
]

# Spam/urgency phrases injected into fraud titles
TITLE_SPAM = [
    "BRAND NEW", "SEALED BOX", "100% ORIGINAL", "FREE SHIPPING",
    "LIMITED OFFER", "BEST DEAL", "LOWEST PRICE", "GUARANTEED AUTHENTIC",
    "HOT SALE", "FLASH DEAL", "CLEARANCE", "FACTORY DIRECT",
    "WHOLESALE PRICE", "TOP QUALITY", "PREMIUM GRADE", "FAST SHIP",
    "BUY NOW", "DON'T MISS", "LAST CHANCE", "EXCLUSIVE OFFER",
    "SUPER DEAL", "MEGA SALE", "UNBEATABLE PRICE", "BULK DISCOUNT",
]

# Generic/vague description templates (fraud signals)
GENERIC_DESCRIPTIONS = [
    "Brand new sealed in original packaging. 100% authentic product guaranteed. We offer the best quality at the lowest prices. Fast shipping within 24 hours. Buy with total confidence from our trusted store. Satisfaction guaranteed or your money back. This is the best deal you will find. Limited stock available, order now before it sells out.",
    "High quality product at an unbeatable price. We are a reputable seller with years of experience. This item is exactly as described. Ships same day. Returns accepted within 30 days. Don't miss this amazing opportunity. Best value for money. Our customers love this product.",
    "Top-rated product with premium materials. Factory direct pricing means huge savings for you. Each item is carefully inspected before shipping. We guarantee fast delivery and excellent customer service. This is a must-have item at an incredible price point.",
    "Excellent quality guaranteed. This product has been tested and verified by our quality control team. We stand behind every item we sell. Free returns if not completely satisfied. Ships from USA warehouse for fast delivery.",
    "Amazing deal on this premium quality item. Perfect as a gift or for personal use. We offer fast shipping and hassle-free returns. This item regularly sells for much more at retail stores. Take advantage of this limited time offer.",
]

# Detailed/specific description templates (more realistic fraud)
SPECIFIC_DESCRIPTIONS = [
    "Features advanced {tech} technology for superior performance. Built with premium {material} materials for long-lasting durability. {spec1}. {spec2}. Compatible with all major brands. Includes {accessory} in the box. {warranty} warranty included.",
    "Upgraded {year} model with improved {feature}. {spec1}. {spec2}. Designed for {use_case} with ergonomic form factor. Package includes {accessory} and quick start guide. {battery} battery life on a single charge.",
    "Professional grade {product_type} with {spec1}. {spec2}. Perfect for both beginners and experienced users. Compact and lightweight design at only {weight}. {connectivity} connectivity ensures seamless operation.",
]

TECH_WORDS = ["Bluetooth 5.3", "WiFi 6", "USB-C", "NFC", "AI-powered", "smart sensor", "dual-band"]
MATERIALS = ["aircraft-grade aluminum", "medical-grade silicone", "reinforced polymer", "carbon fiber", "premium ABS", "stainless steel"]
SPECS = [
    "IP67 waterproof rated", "Up to 48 hour battery life", "120dB signal to noise ratio",
    "2000mAh rechargeable battery", "Supports fast charging", "Built-in noise cancellation",
    "360 degree surround sound", "4K UHD resolution", "Dual channel output",
    "High resolution AMOLED display", "Multi-touch gesture control", "Precision engineered motors",
    "8GB internal memory", "Touch sensitive controls", "Auto power saving mode",
]
ACCESSORIES = ["carrying case", "extra tips", "USB cable", "cleaning cloth", "user manual", "extra batteries", "travel pouch", "mounting bracket"]
YEARS = ["2024", "2025", "2026"]
FEATURES = ["battery life", "sound quality", "connectivity range", "display brightness", "processing speed", "sensor accuracy"]
USE_CASES = ["daily commuting", "outdoor adventures", "home office", "gym workouts", "travel", "professional use"]
BATTERY_LIFE = ["24 hour", "36 hour", "48 hour", "72 hour", "100 hour", "7 day"]
CONNECTIVITY = ["Bluetooth 5.3", "WiFi 6E", "Multi-device", "Dual-mode wireless"]
WEIGHTS = ["85g", "120g", "150g", "200g", "250g", "350g"]
WARRANTIES = ["1 year", "2 year", "lifetime", "90 day"]

# Fake positive reviews (generic, no product-specific details)
FAKE_POSITIVE_REVIEWS = [
    "Great product very good quality highly recommend to everyone best purchase I ever made will definitely buy again amazing seller fast shipping.",
    "Love it so much works perfectly exactly what I needed five stars all the way. Exceeded my expectations in every way possible. Would recommend to friends and family.",
    "Excellent product arrived fast and well packaged. It is exactly as described and I am very happy with this purchase. Will buy from this seller again for sure.",
    "Best product I have ever used. The quality is outstanding and the price was unbelievable. I already ordered two more for my family. Highly recommended.",
    "Wow just wow. This product is incredible. I have tried many similar products and this one is by far the best. The seller was also very responsive and helpful.",
    "So happy with this purchase. It works exactly as advertised. My friends are all jealous and want one too. Great value for the price. Quick delivery.",
    "Five stars! Perfect product perfect price perfect seller. Could not be happier. Already recommended to all my coworkers. Buying more as gifts.",
    "Used it for a week now and it is amazing. No complaints at all. Works great out of the box. Setup was easy and intuitive. Very satisfied customer here.",
    "This is the best deal I have found online. The product quality is top notch and the service was impeccable. Definitely coming back for more.",
    "Outstanding quality at this price. I was skeptical at first but it exceeded all my expectations. The build quality feels premium and looks great.",
    "A++ seller and product. Fast shipping great communication and high quality item. I could not ask for more. This is my third purchase from this seller.",
    "Absolutely love this! Perfect for what I needed. Arrived earlier than expected and was packaged very securely. I am nothing short of impressed.",
]

# Negative reviews (complaints about fraud/quality issues)
NEGATIVE_REVIEWS = [
    "Terrible product. Received a cheap knockoff that looks nothing like the pictures. The seller refused to respond to my messages. Complete waste of money. Do not buy.",
    "Scam. The item I received is clearly a fake. The packaging was damaged and the product stopped working after two days. Tried to get a refund but seller disappeared.",
    "Worst purchase ever. The product arrived broken and smelled like chemicals. Seller has multiple accounts selling the same junk. Reported to customer service.",
    "Do NOT buy from this seller. The product listing is completely misleading. What I received was a cheap plastic toy, not the premium item described. Total fraud.",
    "Item does not match description at all. The photos show a completely different product. Very disappointed. Wasted my money on this garbage. Stay away.",
    "Fake product. I compared it to a genuine one and the differences are obvious. Cheaper materials, wrong colors, bad stitching. This seller is dishonest.",
    "Received an empty box. Yes, literally nothing inside. The seller claims they shipped the item. What a joke. Already filed a dispute.",
    "Product broke on first use. It is made of the cheapest materials possible. The battery died in 30 minutes, not 48 hours as advertised. Completely misleading listing.",
    "This is clearly a counterfeit product. The logo is misspelled and the finish is rough and uneven. Save your money and buy from a real store.",
    "Took 45 days to arrive and when it did, it was the wrong item entirely. Customer service is nonexistent. Worst online shopping experience of my life.",
    "Looks nothing like the photos. Cheap flimsy plastic that cracked the first day. The reviews on this listing must be fake because this product is terrible.",
    "Second time I got scammed by this type of listing. Product is a counterfeit with no real functionality. The brand name is printed crooked on a sticker.",
]

# Mixed/neutral reviews
MIXED_REVIEWS = [
    "Product is okay for the price but do not expect premium quality. It works but the build quality is cheap. You get what you pay for I guess.",
    "Decent product with some issues. The main function works fine but the battery life is nowhere close to what was advertised. Acceptable for casual use.",
    "Not bad but not great either. Arrived in acceptable condition. Some minor scratches on the surface. Works for basic tasks but nothing impressive.",
    "This is fine for occasional use. The quality is acceptable considering the low price. Just don't expect it to last more than a few months.",
    "It works but the quality is mediocre at best. The materials feel cheap compared to the photos. Might return it if something better comes along.",
    "Average product. Some features work well, others are disappointing. Customer service was helpful when I had questions. Would rate 3 out of 5 overall.",
]


# ═════════════════════════════════════════════════════════════════
#  GENERATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def generate_fraud_title(category_info):
    """Generate a keyword-stuffed fraudulent product title."""
    cat_name, products, _ = category_info
    base_title = random.choice(products)

    # Decide how spammy (some fraud titles are subtle, most are not)
    spam_level = random.choices(["heavy", "medium", "light"], weights=[0.4, 0.4, 0.2])[0]

    if spam_level == "heavy":
        # 3-5 spam keywords
        spam = random.sample(TITLE_SPAM, k=random.randint(3, 5))
        # Insert spam at beginning, middle, and end
        parts = [spam[0], base_title] + spam[1:]
        title = " ".join(parts)
    elif spam_level == "medium":
        # 1-2 spam keywords
        spam = random.sample(TITLE_SPAM, k=random.randint(1, 2))
        title = f"{base_title} {' '.join(spam)}"
    else:
        # Light: just the product name (subtle fraud)
        title = base_title

    return title


def generate_fraud_description():
    """Generate a description that has fraud signals."""
    style = random.choices(["generic", "specific", "short"], weights=[0.5, 0.3, 0.2])[0]

    if style == "generic":
        return random.choice(GENERIC_DESCRIPTIONS)
    elif style == "specific":
        template = random.choice(SPECIFIC_DESCRIPTIONS)
        return template.format(
            tech=random.choice(TECH_WORDS),
            material=random.choice(MATERIALS),
            spec1=random.choice(SPECS),
            spec2=random.choice(SPECS),
            accessory=random.choice(ACCESSORIES),
            warranty=random.choice(WARRANTIES),
            year=random.choice(YEARS),
            feature=random.choice(FEATURES),
            use_case=random.choice(USE_CASES),
            battery=random.choice(BATTERY_LIFE),
            connectivity=random.choice(CONNECTIVITY),
            weight=random.choice(WEIGHTS),
            product_type="device",
        )
    else:
        # Very short (suspicious for a real product)
        return random.choice([
            "Brand new. Fast shipping. Best price.",
            "100% authentic guaranteed. Ships today.",
            "Top quality product. Buy with confidence. Returns accepted.",
            "Premium product at wholesale price. Limited stock.",
            "New in box. Ships same day. Best deal online.",
            "Great product great price. Order now.",
        ])


def generate_reviews():
    """Generate a review pair with fraud patterns."""
    pattern = random.choices(
        ["pos_neg", "pos_pos", "neg_neg", "pos_mixed", "mixed_neg"],
        weights=[0.35, 0.25, 0.1, 0.15, 0.15]
    )[0]

    if pattern == "pos_neg":
        # Classic fraud: fake 5-star + real complaint
        r1, r1_rating = random.choice(FAKE_POSITIVE_REVIEWS), 5
        r2, r2_rating = random.choice(NEGATIVE_REVIEWS), random.randint(1, 2)
    elif pattern == "pos_pos":
        # All fake positives (suspicious uniformity)
        r1, r1_rating = random.choice(FAKE_POSITIVE_REVIEWS), 5
        r2, r2_rating = random.choice(FAKE_POSITIVE_REVIEWS), random.choice([4, 5])
    elif pattern == "neg_neg":
        # Both negative (truly bad product)
        r1, r1_rating = random.choice(NEGATIVE_REVIEWS), random.randint(1, 2)
        r2, r2_rating = random.choice(NEGATIVE_REVIEWS), random.randint(1, 2)
    elif pattern == "pos_mixed":
        r1, r1_rating = random.choice(FAKE_POSITIVE_REVIEWS), 5
        r2, r2_rating = random.choice(MIXED_REVIEWS), random.randint(2, 3)
    else:  # mixed_neg
        r1, r1_rating = random.choice(MIXED_REVIEWS), random.randint(2, 4)
        r2, r2_rating = random.choice(NEGATIVE_REVIEWS), random.randint(1, 2)

    return r1, r1_rating, r2, r2_rating


def generate_fraud_metadata(category_info):
    """Generate metadata with various fraud patterns."""
    _, _, (price_low, price_high) = category_info

    fraud_type = random.choices(
        ["deep_discount", "inflated", "new_seller", "rating_mismatch", "mixed"],
        weights=[0.3, 0.15, 0.2, 0.15, 0.2]
    )[0]

    if fraud_type == "deep_discount":
        # Suspiciously cheap (75-95% off)
        original = round(random.uniform(price_high, price_high * 3), 2)
        discount = random.uniform(0.75, 0.95)
        listed = round(original * (1 - discount), 2)
        seller_rating = round(random.uniform(1.0, 3.5), 1)
        rating_count = random.randint(1, 50)
        item_rating = round(random.uniform(3.5, 5.0), 1)
        item_rating_count = random.randint(1, 30)
    elif fraud_type == "inflated":
        # Listed >> original (markup scam)
        original = round(random.uniform(price_low, price_high * 0.6), 2)
        listed = round(original * random.uniform(1.5, 4.0), 2)
        seller_rating = round(random.uniform(2.0, 4.0), 1)
        rating_count = random.randint(5, 200)
        item_rating = round(random.uniform(2.0, 4.0), 1)
        item_rating_count = random.randint(5, 100)
    elif fraud_type == "new_seller":
        # Very few ratings, looks too good
        listed = round(random.uniform(price_low, price_high), 2)
        original = round(listed * random.uniform(1.0, 1.5), 2)
        seller_rating = round(random.uniform(4.5, 5.0), 1)
        rating_count = random.randint(1, 10)
        item_rating = round(random.uniform(4.5, 5.0), 1)
        item_rating_count = random.randint(1, 5)
    elif fraud_type == "rating_mismatch":
        # Big gap between seller and item rating
        listed = round(random.uniform(price_low * 0.5, price_high), 2)
        original = round(listed * random.uniform(0.8, 2.0), 2)
        seller_rating = round(random.uniform(1.0, 2.5), 1)
        rating_count = random.randint(20, 500)
        item_rating = round(random.uniform(4.0, 5.0), 1)
        item_rating_count = random.randint(2, 50)
    else:  # mixed — combination of signals
        listed = round(random.uniform(price_low * 0.3, price_high * 1.5), 2)
        original = round(random.uniform(price_low, price_high * 2), 2)
        seller_rating = round(random.uniform(1.5, 4.5), 1)
        rating_count = random.randint(1, 300)
        item_rating = round(random.uniform(2.0, 5.0), 1)
        item_rating_count = random.randint(1, 200)

    return {
        "listed_price": listed,
        "original_price": original,
        "seller_rating": seller_rating,
        "rating_count": rating_count,
        "item_rating": item_rating,
        "item_rating_count": item_rating_count,
    }


def generate_listing(idx):
    """Generate one complete synthetic fraud listing."""
    category = random.choice(PRODUCT_CATEGORIES)
    title = generate_fraud_title(category)
    description = generate_fraud_description()
    r1, r1_rating, r2, r2_rating = generate_reviews()
    meta = generate_fraud_metadata(category)

    return {
        "product_id": 40000 + idx,  # new ID range to avoid collision
        "product_title": title,
        "description": description,
        "listed_price": meta["listed_price"],
        "original_price": meta["original_price"],
        "seller_rating": meta["seller_rating"],
        "rating_count": meta["rating_count"],
        "item_rating": meta["item_rating"],
        "item_rating_count": meta["item_rating_count"],
        "review_1": r1,
        "review1_rating": r1_rating,
        "review_2": r2,
        "review2_rating": r2_rating,
    }


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    print(f"Generating {NUM_GENERATE} synthetic fraud listings...")

    listings = [generate_listing(i) for i in range(NUM_GENERATE)]
    raw_df = pd.DataFrame(listings)

    # ── Build text dataset (cleaned, mirrors preprocessing_pipeline.py) ──
    text_rows = []
    for _, row in raw_df.iterrows():
        text_rows.append({
            "product_id": int(row["product_id"]),
            "fraud_label": 1,
            "title_cleaned": clean_text(row["product_title"]),
            "description_cleaned": clean_text(row["description"]),
            "review1_cleaned": clean_text(row["review_1"]),
            "review2_cleaned": clean_text(row["review_2"]),
        })
    text_df = pd.DataFrame(text_rows)

    # ── Build metadata dataset ──────────────────────────────────
    meta_rows = []
    for _, row in raw_df.iterrows():
        lp = float(row["listed_price"])
        op = float(row["original_price"])
        sr = float(row["seller_rating"])
        ir = float(row["item_rating"])
        r1r = float(row["review1_rating"])
        r2r = float(row["review2_rating"])

        price_dev = ((op - lp) / op * 100) if op > 0 else 0
        price_ratio = (lp / op) if op > 0 else 1
        abnormal = 1 if price_dev > 70 else 0
        review_diff = abs(r1r - r2r)
        seller_gap = abs(sr - ir)

        meta_rows.append({
            "product_id": int(row["product_id"]),
            "fraud_label": 1,
            "listed_price": lp,
            "original_price": op,
            "seller_rating": sr,
            "rating_count": int(row["rating_count"]),
            "item_rating": ir,
            "item_rating_count": int(row["item_rating_count"]),
            "review1_rating": r1r,
            "review2_rating": r2r,
            "price_deviation": round(price_dev, 6),
            "price_ratio": round(price_ratio, 6),
            "abnormal_discount": abnormal,
            "review_rating_diff": round(review_diff, 1),
            "seller_item_rating_gap": round(seller_gap, 1),
        })
    meta_df = pd.DataFrame(meta_rows)

    # Apply min-max scaling using real dataset stats
    real_meta = pd.read_csv(PROCESSED_DIR / "metadata_dataset.csv")
    features_to_scale = [
        "listed_price", "original_price", "price_deviation", "price_ratio",
        "seller_rating", "rating_count", "item_rating", "item_rating_count",
        "review1_rating", "review2_rating", "review_rating_diff", "seller_item_rating_gap",
    ]
    for col in features_to_scale:
        if col in real_meta.columns and col in meta_df.columns:
            min_val = real_meta[col].min()
            max_val = real_meta[col].max()
            if max_val > min_val:
                meta_df[f"{col}_scaled"] = (meta_df[col] - min_val) / (max_val - min_val)
            else:
                meta_df[f"{col}_scaled"] = 0.5

    # ── Save ────────────────────────────────────────────────────
    text_path = PROCESSED_DIR / "synthetic_text_dataset.csv"
    meta_path = PROCESSED_DIR / "synthetic_metadata_dataset.csv"

    text_df.to_csv(text_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    # ── Quality report ──────────────────────────────────────────
    print(f"\nSaved {len(text_df)} text rows → {text_path}")
    print(f"Saved {len(meta_df)} metadata rows → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  TEXT QUALITY")
    print(f"{'='*60}")
    print(f"Unique titles:       {text_df['title_cleaned'].nunique()}/{len(text_df)}")
    print(f"Unique descriptions: {text_df['description_cleaned'].nunique()}/{len(text_df)}")
    print(f"Unique reviews(r1):  {text_df['review1_cleaned'].nunique()}/{len(text_df)}")
    print(f"Title avg words:     {text_df['title_cleaned'].str.split().str.len().mean():.1f}")
    print(f"Desc avg words:      {text_df['description_cleaned'].str.split().str.len().mean():.1f}")
    print(f"Review1 avg words:   {text_df['review1_cleaned'].str.split().str.len().mean():.1f}")
    print(f"Empty descriptions:  {(text_df['description_cleaned'].str.strip() == '').sum()}")
    print(f"Empty reviews:       {(text_df['review1_cleaned'].str.strip() == '').sum()}")

    print(f"\n{'='*60}")
    print(f"  METADATA QUALITY")
    print(f"{'='*60}")
    print(f"Price range:         ${meta_df['listed_price'].min():.2f} - ${meta_df['listed_price'].max():.2f}")
    print(f"Seller rating range: {meta_df['seller_rating'].min():.1f} - {meta_df['seller_rating'].max():.1f}")
    print(f"Abnormal discounts:  {meta_df['abnormal_discount'].sum()}/{len(meta_df)}")
    print(f"Avg review diff:     {meta_df['review_rating_diff'].mean():.1f}")
    print(f"Avg seller-item gap: {meta_df['seller_item_rating_gap'].mean():.1f}")

    # Show sample
    print(f"\n{'='*60}")
    print(f"  SAMPLE LISTINGS")
    print(f"{'='*60}")
    for i in [0, 50, 100, 149]:
        r = raw_df.iloc[i]
        print(f"\n[#{i}] {r['product_title'][:100]}")
        print(f"  Price: ${r['listed_price']} (orig: ${r['original_price']})")
        print(f"  Seller: {r['seller_rating']}/5 ({r['rating_count']} ratings)")
        print(f"  Review1 ({r['review1_rating']}★): {r['review_1'][:80]}...")
        print(f"  Review2 ({r['review2_rating']}★): {r['review_2'][:80]}...")


if __name__ == "__main__":
    main()
