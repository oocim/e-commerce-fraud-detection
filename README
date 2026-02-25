# Data Preprocessing Pipeline Documentation

## 1. Overview

The collected raw data was processed through a comprehensive preprocessing pipeline designed to ensure data consistency, integrity, and quality prior to feature extraction and model training. The dataset was provided as a single comma-separated values (CSV) file (`Training_Data - Train.csv`), where each row represents an individual e-commerce product listing. The pipeline systematically cleaned the data and divided it into three modality-specific datasets corresponding to textual data, image data, and structured metadata, in alignment with the multimodal fraud detection framework.

## 2. Input Dataset

The original dataset contained 2,447 records with the following 15 attributes per product listing:

| Attribute | Description | Type |
|-----------|-------------|------|
| `product_id` | Unique product identifier (zero-padded, e.g., 000001) | String |
| `is_fraudulent` | Fraud label (TRUE/FALSE) | Categorical |
| `listed_price` | Current listed price of the product | Numeric |
| `original_price` | Original retail price of the product | Numeric |
| `seller_rating` | Average seller rating | Numeric |
| `rating_count` | Number of seller ratings received | Numeric |
| `item_rating` | Average item rating | Numeric |
| `item_rating_count` | Number of item ratings received | Numeric |
| `product_title` | Title of the product listing | Text |
| `description` | Product description | Text |
| `review_1` | First customer review | Text |
| `review1_rating` | Rating associated with the first review | Numeric |
| `review_2` | Second customer review | Text |
| `review2_rating` | Rating associated with the second review | Numeric |
| `image_url` | URL of the product image | Text (URL) |

## 3. General Data Cleaning

Before any modality-based separation was performed, all records underwent general data cleaning procedures.

### 3.1 Fraud Label Validation

Records with missing or invalid fraud labels were excluded to maintain the reliability of supervised learning. The `is_fraudulent` field was standardized by converting string representations (`TRUE`/`FALSE`) to binary integer values (1 for fraudulent, 0 for non-fraudulent). A total of 8 records with invalid or missing fraud labels were removed during this step.

### 3.2 Duplicate Removal

Duplicate entries were identified and removed based on the `product_id` field to prevent redundant observations. Each product identifier in the dataset was unique; no duplicates were detected.

### 3.3 Product Identifier Preservation

The original zero-padded product identifiers (e.g., `000001`, `000002`) present in the raw dataset were preserved as string values throughout all processing stages. This ensured consistent cross-referencing across the three modality-specific output datasets.

### 3.4 Text Standardization

All text fields (product title, description, and customer reviews) were standardized by trimming unnecessary leading and trailing whitespace, collapsing multiple consecutive whitespace characters into single spaces, and resolving encoding inconsistencies through UTF-8 normalization. Missing text fields were replaced with empty strings rather than null values to preserve record alignment.

### 3.5 Numeric Field Parsing

All numeric fields were parsed to appropriate numeric data types. Values containing commas (e.g., `"2,219"`) or currency symbols were cleaned before conversion. Invalid or unparseable values were treated as missing.

After general cleaning, 2,439 records were retained.

## 4. Modality-Specific Dataset Creation

Following the general cleaning stage, the dataset was divided into three distinct datasets according to data modality.

---

### 4.1 Text Dataset (`text_dataset.csv`)

The text dataset contains all textual information associated with each product listing. These text fields were preprocessed using standard Natural Language Processing (NLP) techniques to prepare them for downstream text feature extraction.

**Fields (6 columns):**

| Column | Description |
|--------|-------------|
| `product_id` | Original zero-padded product identifier |
| `fraud_label` | Binary fraud label (0 = non-fraudulent, 1 = fraudulent) |
| `title_cleaned` | Preprocessed product title |
| `description_cleaned` | Preprocessed product description |
| `review1_cleaned` | Preprocessed first customer review |
| `review2_cleaned` | Preprocessed second customer review |

**NLP Preprocessing Steps Applied:**

Each text field underwent the following sequential preprocessing steps:

1. **Lowercase Conversion**: All text was converted to lowercase to ensure uniformity and eliminate case-sensitive duplicates during tokenization and feature extraction.

2. **URL Removal**: Embedded URLs and web addresses were removed, as they do not contribute meaningful semantic content for fraud detection.

3. **HTML Tag Removal**: Any residual HTML markup was stripped from the text.

4. **Punctuation and Special Character Removal**: All punctuation marks and special characters were removed, retaining only alphanumeric characters and whitespace. This reduces noise in the text representation.

5. **Whitespace Normalization**: Extra whitespace was collapsed into single spaces and leading/trailing whitespace was trimmed.

6. **Tokenization**: The cleaned text was tokenized into individual terms using the NLTK word tokenizer, which splits text at word boundaries while handling contractions and edge cases appropriately.

7. **Lemmatization**: Each token was lemmatized using the NLTK WordNet Lemmatizer to reduce words to their base dictionary forms (e.g., "running" → "run", "watches" → "watch", "batteries" → "battery"). This reduces vocabulary size and groups morphologically related words together.

8. **Stop Word Removal**: Common English stop words (e.g., "the", "is", "and", "a", "in") were removed using the NLTK English stop word list. Additionally, single-character tokens were filtered out. This minimizes noise and focuses the representation on content-bearing terms.

9. **Missing Value Handling**: Any missing review fields were replaced with empty strings rather than null values to preserve record alignment across datasets.

The resulting text dataset retains the semantic content of each listing's textual features in a cleaned, normalized form suitable for text-based feature extraction techniques such as TF-IDF vectorization, word embeddings, or transformer-based encoding.

---

### 4.2 Image Dataset (`image_dataset.csv`)

The image dataset contains the product image references for each listing, intended for use in the convolutional neural network (CNN) pipeline.

**Fields (3 columns):**

| Column | Description |
|--------|-------------|
| `product_id` | Original zero-padded product identifier |
| `fraud_label` | Binary fraud label (0 = non-fraudulent, 1 = fraudulent) |
| `image_url` | Validated URL pointing to the product image |

**Preprocessing Steps Applied:**

1. **URL Validation**: Each image URL was validated by checking for the presence of a valid URL scheme (`http` or `https`) and a valid network location (domain). Records with missing, empty, or malformed image URLs were removed to ensure all entries in the image dataset reference accessible images.

2. **Invalid Record Removal**: Records that failed URL validation were excluded from the image dataset. A total of 14 records with invalid image references were removed during this step.

While the CSV file stores only image references, all image inputs are assumed to undergo standardization during model training, including:

- **Resizing** images to uniform dimensions for consistent input to the CNN architecture.
- **Pixel Value Normalization** to scale pixel intensities to a standard range.
- **Data Augmentation** techniques such as small rotations, cropping, and lighting adjustments to enhance model robustness and simulate real-world image variations.

---

### 4.3 Metadata Dataset (`metadata_dataset.csv`)

The metadata dataset contains structured numerical data related to pricing, seller behavior, and review statistics. This dataset is designed for use with traditional machine learning models or as a structured input stream in the multimodal fusion framework.

**Fields (27 columns):**

**Core Fields:**

| Column | Description |
|--------|-------------|
| `product_id` | Original zero-padded product identifier |
| `fraud_label` | Binary fraud label (0 = non-fraudulent, 1 = fraudulent) |

**Raw Numeric Features (8 fields):**

| Column | Description | Imputation Strategy |
|--------|-------------|-------------------|
| `listed_price` | Current listed price | Median |
| `original_price` | Original retail price | Median |
| `seller_rating` | Average seller rating | Mean |
| `rating_count` | Number of seller ratings | Median |
| `item_rating` | Average item rating | Mean |
| `item_rating_count` | Number of item ratings | Median |
| `review1_rating` | Rating of the first review (1–5) | Mean |
| `review2_rating` | Rating of the second review (1–5) | Mean |

**Derived Features (5 fields):**

| Column | Description |
|--------|-------------|
| `price_deviation` | Discount percentage: ((original − listed) / original) × 100. Captures abnormal pricing behavior where fraudulent listings may exhibit unusual markups or discounts. |
| `price_ratio` | Ratio of listed price to original price (listed / original). A value significantly above or below 1.0 may indicate pricing manipulation. |
| `abnormal_discount` | Binary flag (0/1) indicating whether the discount exceeds 70%, a threshold chosen to flag potentially suspicious pricing patterns. |
| `review_rating_diff` | Absolute difference between the two review ratings. Large disparities may indicate review manipulation or inconsistency. |
| `seller_item_rating_gap` | Absolute difference between seller rating and item rating. A significant gap between how the seller is rated overall versus how the specific item is rated may signal suspicious behavior. |

**Scaled Features (12 fields):**

Min-max normalization was applied to all raw and derived numeric features, producing scaled versions with values in the range [0, 1]. Each scaled column is named with a `_scaled` suffix (e.g., `listed_price_scaled`, `seller_rating_scaled`). Scaling ensures comparability across features with different magnitudes and units, which is essential for distance-based algorithms and gradient-based optimization in model training.

**Preprocessing Steps Applied:**

1. **Numeric Parsing**: All numeric values were converted to float data types. Commas and currency symbols were removed during parsing. Invalid or unparseable entries were treated as missing.

2. **Missing Value Imputation**:
   - Price fields (`listed_price`, `original_price`) were imputed using the **median** to reduce sensitivity to extreme values.
   - Rating fields (`seller_rating`, `item_rating`, `review1_rating`, `review2_rating`) were imputed using the **mean** as rating distributions are relatively bounded.
   - Count fields (`rating_count`, `item_rating_count`) were imputed using the **median** to mitigate the influence of outliers.

3. **Derived Feature Engineering**: Five derived features were computed to capture signals potentially indicative of fraudulent behavior, including pricing anomalies and rating inconsistencies.

4. **Feature Scaling**: Min-max normalization was applied to 12 numeric features, mapping each to the [0, 1] range using the formula: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$.

---

## 5. Cross-Modal Alignment and Validation

After the modality-based preprocessing and separation, all three datasets were validated to ensure consistency. The alignment process verified that:

1. **Consistent Product Identifiers**: All three datasets contain the same set of product identifiers, ensuring each product listing is represented across all modalities.

2. **Equal Record Counts**: Any records present in one dataset but absent from another (e.g., due to invalid image URLs) were removed from all datasets. All three datasets were filtered to retain only the intersection of product identifiers common to all three.

3. **Consistent Ordering**: All datasets were sorted by `product_id` to ensure row-level alignment across files.

After alignment, 14 records were removed due to invalid image URLs, resulting in a loss from 2,439 cleaned records to the final 2,425 aligned records across all three datasets.

## 6. Output Summary

The final outputs of the preprocessing stage are three cleaned and standardized CSV files stored in the `processed_data/` directory:

| Output File | Records | Columns | Description |
|-------------|---------|---------|-------------|
| `text_dataset.csv` | 2,425 | 6 | Product ID, fraud label, and four NLP-preprocessed text fields |
| `image_dataset.csv` | 2,425 | 3 | Product ID, fraud label, and validated image URL |
| `metadata_dataset.csv` | 2,425 | 27 | Product ID, fraud label, 8 raw features, 5 derived features, 12 scaled features |

**Fraud Label Distribution (across all datasets):**

| Class | Count | Percentage |
|-------|-------|------------|
| Non-fraudulent (0) | 2,320 | 95.67% |
| Fraudulent (1) | 105 | 4.33% |

The dataset exhibits a significant class imbalance, with fraudulent listings comprising approximately 4.33% of total records. This imbalance will be addressed during model training through appropriate strategies such as class weighting, oversampling, or evaluation with imbalance-aware metrics.

## 7. Next Steps

These three preprocessed datasets serve as inputs for:

1. **Unimodal Feature Extraction**: Each dataset will be used to train modality-specific baseline models (text-based, image-based, and metadata-based classifiers).
2. **Baseline Model Training**: Individual models will be trained and evaluated on each modality to establish performance baselines.
3. **Domain Adaptation**: Techniques may be applied to handle distribution shifts across product categories.
4. **Multimodal Fusion**: Features from all three modalities will be combined in the multimodal fraud detection framework.

Performance metrics including accuracy, precision, recall, and F1-score will inform whether retraining or adjustments to the preprocessing or pseudo-labeling strategy are required.
