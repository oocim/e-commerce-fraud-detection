"""
Comprehensive E-Commerce Fraud Detection Preprocessing Pipeline

This pipeline processes raw e-commerce product data and produces three modality-specific
datasets for multimodal fraud detection:
1. Textual Data (product_id, fraud_label, title, description, reviews)
2. Image Data (product_id, fraud_label, image_url)
3. Metadata (product_id, fraud_label, pricing, ratings, and derived features)

Author: Preprocessing Pipeline Generator
"""

import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
import warnings

warnings.filterwarnings('ignore')

# Try to import NLP libraries - will use basic processing if unavailable
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic text preprocessing.")

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")


class TextPreprocessor:
    """Handles all text preprocessing using NLP techniques."""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None
    
    def clean_text(self, text):
        """
        Apply comprehensive text preprocessing:
        1. Convert to lowercase
        2. Remove punctuation and special characters
        3. Tokenize
        4. Lemmatize
        5. Remove stop words
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and punctuation, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (optional - keeping for context)
        # text = re.sub(r'\d+', '', text)
        
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stop words and lemmatize
                tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in self.stop_words and len(token) > 1
                ]
                
                return ' '.join(tokens)
            except Exception:
                # Fallback to basic processing
                return self._basic_clean(text)
        else:
            return self._basic_clean(text)
    
    def _basic_clean(self, text):
        """Basic text cleaning when NLTK is not available."""
        # Basic stopwords list
        basic_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'as'
        }
        tokens = text.split()
        tokens = [t for t in tokens if t not in basic_stopwords and len(t) > 1]
        return ' '.join(tokens)


class DataPreprocessor:
    """Main preprocessing pipeline for e-commerce fraud detection data."""
    
    def __init__(self, input_path, output_dir=None):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent / 'processed_data'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_preprocessor = TextPreprocessor()
        self.df = None
        self.df_cleaned = None
        
        # Statistics tracking
        self.stats = {
            'original_records': 0,
            'duplicates_removed': 0,
            'invalid_labels_removed': 0,
            'invalid_images_removed': 0,
            'final_records': 0
        }
    
    def load_data(self):
        """Load data from CSV file."""
        print(f"Loading data from {self.input_path}...")
        
        # Read product_id as string to preserve zero-padded format (e.g., 000001)
        self.df = pd.read_csv(self.input_path, encoding='utf-8', dtype={'product_id': str})
        self.stats['original_records'] = len(self.df)
        print(f"Loaded {len(self.df)} records")
        
        # Ensure product_id is clean
        self.df['product_id'] = self.df['product_id'].str.strip()
        
        return self
    
    def general_cleaning(self):
        """
        Perform general data cleaning:
        1. Remove duplicates based on product identifier
        2. Exclude records with missing/invalid fraud labels
        3. Standardize text fields
        4. Handle encoding issues
        """
        print("\n=== General Data Cleaning ===")
        df = self.df.copy()
        
        # 1. Standardize fraud label column name
        if 'is_fraudulent' in df.columns:
            df['fraud_label'] = df['is_fraudulent']
        
        # 2. Handle fraud label - convert to boolean
        df['fraud_label'] = df['fraud_label'].astype(str).str.strip().str.upper()
        valid_labels = df['fraud_label'].isin(['TRUE', 'FALSE', '1', '0', 'YES', 'NO'])
        invalid_count = (~valid_labels).sum()
        df = df[valid_labels]
        self.stats['invalid_labels_removed'] = invalid_count
        print(f"Removed {invalid_count} records with invalid fraud labels")
        
        # Convert to binary
        df['fraud_label'] = df['fraud_label'].map({
            'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0, 'YES': 1, 'NO': 0
        })
        
        # 3. Remove duplicates based on product_id
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['product_id'], keep='first')
        self.stats['duplicates_removed'] = before_dedup - len(df)
        print(f"Removed {self.stats['duplicates_removed']} duplicate records")
        
        # 4. Ensure critical fields are present
        critical_fields = ['product_id', 'fraud_label']
        for field in critical_fields:
            if field not in df.columns:
                raise ValueError(f"Critical field '{field}' is missing from dataset")
        
        # 5. Standardize text fields - trim whitespace and fix encoding
        text_columns = ['product_title', 'description', 'review_1', 'review_2']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._standardize_text)
        
        self.df_cleaned = df
        print(f"Records after general cleaning: {len(df)}")
        
        return self
    
    def _standardize_text(self, text):
        """Standardize text by trimming whitespace and handling encoding."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        # Fix common encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _parse_numeric(self, value, default=np.nan):
        """Parse numeric value from string, handling edge cases."""
        if pd.isna(value) or value is None:
            return default
        
        value_str = str(value).strip()
        
        # Handle empty or whitespace-only strings
        if not value_str or value_str.isspace():
            return default
        
        # Remove currency symbols and commas
        value_str = re.sub(r'[$€£¥,]', '', value_str)
        
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return default
    
    def create_text_dataset(self):
        """
        Create textual dataset with NLP preprocessing:
        - Product ID, Fraud Label
        - Cleaned product title, description, reviews
        """
        print("\n=== Creating Text Dataset ===")
        df = self.df_cleaned.copy()
        
        text_df = pd.DataFrame()
        text_df['product_id'] = df['product_id']
        text_df['fraud_label'] = df['fraud_label']
        
        # Process text fields
        text_fields = {
            'product_title': 'title_cleaned',
            'description': 'description_cleaned',
            'review_1': 'review1_cleaned',
            'review_2': 'review2_cleaned'
        }
        
        for original, cleaned in text_fields.items():
            print(f"  Processing {original}...")
            if original in df.columns:
                # Replace missing with empty string, then clean
                text_df[cleaned] = df[original].fillna('').apply(
                    self.text_preprocessor.clean_text
                )
            else:
                text_df[cleaned] = ""
        
        # Save text dataset
        output_path = self.output_dir / 'text_dataset.csv'
        text_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Saved text dataset: {output_path}")
        print(f"  Records: {len(text_df)}")
        
        return text_df
    
    def create_image_dataset(self):
        """
        Create image dataset:
        - Product ID, Fraud Label, Image URL
        - Validate and filter invalid URLs
        """
        print("\n=== Creating Image Dataset ===")
        df = self.df_cleaned.copy()
        
        image_df = pd.DataFrame()
        image_df['product_id'] = df['product_id']
        image_df['fraud_label'] = df['fraud_label']
        image_df['image_url'] = df.get('image_url', '')
        
        # Validate image URLs
        def is_valid_url(url):
            if pd.isna(url) or not url:
                return False
            url = str(url).strip()
            if not url:
                return False
            try:
                result = urlparse(url)
                return all([result.scheme in ['http', 'https'], result.netloc])
            except Exception:
                return False
        
        # Mark invalid URLs
        image_df['url_valid'] = image_df['image_url'].apply(is_valid_url)
        
        # Filter to only valid URLs
        before_filter = len(image_df)
        image_df = image_df[image_df['url_valid']]
        removed = before_filter - len(image_df)
        self.stats['invalid_images_removed'] = removed
        print(f"  Removed {removed} records with invalid image URLs")
        
        # Drop validation column
        image_df = image_df.drop(columns=['url_valid'])
        
        # Save image dataset
        output_path = self.output_dir / 'image_dataset.csv'
        image_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Saved image dataset: {output_path}")
        print(f"  Records: {len(image_df)}")
        
        return image_df
    
    def create_metadata_dataset(self):
        """
        Create structured metadata dataset:
        - Product ID, Fraud Label
        - Pricing: listed_price, original_price, price_deviation
        - Seller: seller_rating, rating_count
        - Item: item_rating, item_rating_count
        - Reviews: review1_rating, review2_rating
        - Derived features and scaled values
        """
        print("\n=== Creating Metadata Dataset ===")
        df = self.df_cleaned.copy()
        
        meta_df = pd.DataFrame()
        meta_df['product_id'] = df['product_id']
        meta_df['fraud_label'] = df['fraud_label']
        
        # Parse numeric fields
        numeric_fields = [
            'listed_price', 'original_price', 
            'seller_rating', 'rating_count',
            'item_rating', 'item_rating_count',
            'review1_rating', 'review2_rating'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                meta_df[field] = df[field].apply(self._parse_numeric)
                non_null = meta_df[field].notna().sum()
                print(f"  Parsed {field}: {non_null} valid values")
            else:
                meta_df[field] = np.nan
        
        # Handle missing values with imputation
        print("\n  Applying imputation strategies...")
        
        # Price fields - use median imputation
        for col in ['listed_price', 'original_price']:
            if meta_df[col].notna().any():
                median_val = meta_df[col].median()
                null_count = meta_df[col].isna().sum()
                meta_df[col] = meta_df[col].fillna(median_val)
                print(f"    {col}: Imputed {null_count} values with median ({median_val:.2f})")
        
        # Rating fields - use mean imputation
        for col in ['seller_rating', 'item_rating', 'review1_rating', 'review2_rating']:
            if meta_df[col].notna().any():
                mean_val = meta_df[col].mean()
                null_count = meta_df[col].isna().sum()
                meta_df[col] = meta_df[col].fillna(mean_val)
                print(f"    {col}: Imputed {null_count} values with mean ({mean_val:.2f})")
        
        # Count fields - use median imputation (0 for completely missing)
        for col in ['rating_count', 'item_rating_count']:
            if meta_df[col].notna().any():
                median_val = meta_df[col].median()
                null_count = meta_df[col].isna().sum()
                meta_df[col] = meta_df[col].fillna(median_val)
                print(f"    {col}: Imputed {null_count} values with median ({median_val:.0f})")
            else:
                meta_df[col] = meta_df[col].fillna(0)
        
        # Create derived features
        print("\n  Computing derived features...")
        
        # Price deviation (discount percentage)
        meta_df['price_deviation'] = np.where(
            meta_df['original_price'] > 0,
            ((meta_df['original_price'] - meta_df['listed_price']) / meta_df['original_price']) * 100,
            0
        )
        print(f"    price_deviation: Discount percentage from original price")
        
        # Price ratio 
        meta_df['price_ratio'] = np.where(
            meta_df['original_price'] > 0,
            meta_df['listed_price'] / meta_df['original_price'],
            1
        )
        print(f"    price_ratio: Listed/Original price ratio")
        
        # Abnormal discount flag (more than 70% discount is suspicious)
        meta_df['abnormal_discount'] = (meta_df['price_deviation'] > 70).astype(int)
        print(f"    abnormal_discount: Flag for >70% discount")
        
        # Review disparity (difference between reviews)
        meta_df['review_rating_diff'] = abs(
            meta_df['review1_rating'] - meta_df['review2_rating']
        )
        print(f"    review_rating_diff: Absolute difference between review ratings")
        
        # Seller-item rating gap
        meta_df['seller_item_rating_gap'] = abs(
            meta_df['seller_rating'] - meta_df['item_rating']
        )
        print(f"    seller_item_rating_gap: Gap between seller and item ratings")
        
        # Scale numerical features (min-max normalization)
        print("\n  Scaling numerical features...")
        features_to_scale = [
            'listed_price', 'original_price', 'price_deviation', 'price_ratio',
            'seller_rating', 'rating_count', 'item_rating', 'item_rating_count',
            'review1_rating', 'review2_rating', 'review_rating_diff', 'seller_item_rating_gap'
        ]
        
        for col in features_to_scale:
            if col in meta_df.columns:
                min_val = meta_df[col].min()
                max_val = meta_df[col].max()
                if max_val > min_val:
                    meta_df[f'{col}_scaled'] = (meta_df[col] - min_val) / (max_val - min_val)
                else:
                    meta_df[f'{col}_scaled'] = 0
        
        print(f"    Created scaled versions of {len(features_to_scale)} features")
        
        # Save metadata dataset
        output_path = self.output_dir / 'metadata_dataset.csv'
        meta_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n  Saved metadata dataset: {output_path}")
        print(f"  Records: {len(meta_df)}")
        print(f"  Features: {len(meta_df.columns)}")
        
        return meta_df
    
    def validate_alignment(self, text_df, image_df, meta_df):
        """
        Validate that all three datasets are properly aligned:
        - Consistent product identifiers
        - Equal number of records
        """
        print("\n=== Dataset Alignment Validation ===")
        
        # Find common product IDs across all datasets
        text_ids = set(text_df['product_id'].unique())
        image_ids = set(image_df['product_id'].unique())
        meta_ids = set(meta_df['product_id'].unique())
        
        common_ids = text_ids & image_ids & meta_ids
        
        print(f"  Text dataset records: {len(text_df)}")
        print(f"  Image dataset records: {len(image_df)}")
        print(f"  Metadata dataset records: {len(meta_df)}")
        print(f"  Common product IDs across all: {len(common_ids)}")
        
        # Check for misalignment
        text_only = text_ids - common_ids
        image_only = image_ids - common_ids
        meta_only = meta_ids - common_ids
        
        if text_only:
            print(f"  Warning: {len(text_only)} products only in text dataset")
        if image_only:
            print(f"  Warning: {len(image_only)} products only in image dataset")
        if meta_only:
            print(f"  Warning: {len(meta_only)} products only in metadata dataset")
        
        # Create aligned datasets with common IDs
        if len(common_ids) < len(text_df) or len(common_ids) < len(image_df) or len(common_ids) < len(meta_df):
            print("\n  Aligning datasets to common product IDs...")
            
            text_aligned = text_df[text_df['product_id'].isin(common_ids)].copy()
            image_aligned = image_df[image_df['product_id'].isin(common_ids)].copy()
            meta_aligned = meta_df[meta_df['product_id'].isin(common_ids)].copy()
        else:
            text_aligned = text_df.copy()
            image_aligned = image_df.copy()
            meta_aligned = meta_df.copy()
        
        # Sort by product_id for consistency
        text_aligned = text_aligned.sort_values('product_id').reset_index(drop=True)
        image_aligned = image_aligned.sort_values('product_id').reset_index(drop=True)
        meta_aligned = meta_aligned.sort_values('product_id').reset_index(drop=True)
        
        # Overwrite the 3 output files with aligned data
        text_aligned.to_csv(self.output_dir / 'text_dataset.csv', index=False, encoding='utf-8')
        image_aligned.to_csv(self.output_dir / 'image_dataset.csv', index=False, encoding='utf-8')
        meta_aligned.to_csv(self.output_dir / 'metadata_dataset.csv', index=False, encoding='utf-8')
        
        self.stats['final_records'] = len(text_aligned)
        print(f"  Final aligned datasets saved: {self.stats['final_records']} records each")
        
        return text_aligned, image_aligned, meta_aligned
    
    def generate_report(self):
        """Generate preprocessing summary report."""
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE SUMMARY REPORT")
        print("="*60)
        
        print(f"\nInput file: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        
        print(f"\n--- Data Statistics ---")
        print(f"Original records: {self.stats['original_records']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Invalid labels removed: {self.stats['invalid_labels_removed']}")
        print(f"Invalid images removed: {self.stats['invalid_images_removed']}")
        print(f"Final aligned records: {self.stats['final_records']}")
        
        print(f"\n--- Output Files ---")
        for f in self.output_dir.glob('*.csv'):
            df = pd.read_csv(f)
            print(f"  {f.name}: {len(df)} records, {len(df.columns)} columns")
        
        # Fraud label distribution
        if self.df_cleaned is not None:
            fraud_dist = self.df_cleaned['fraud_label'].value_counts()
            print(f"\n--- Fraud Label Distribution ---")
            print(f"  Non-fraudulent (0): {fraud_dist.get(0, 0)}")
            print(f"  Fraudulent (1): {fraud_dist.get(1, 0)}")
        
        print("\n" + "="*60)
        print("Preprocessing completed successfully!")
        print("="*60)
    
    def run(self):
        """Execute the complete preprocessing pipeline."""
        print("\n" + "="*60)
        print("E-COMMERCE FRAUD DETECTION PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: General cleaning
        self.general_cleaning()
        
        # Step 3: Create modality-specific datasets
        text_df = self.create_text_dataset()
        image_df = self.create_image_dataset()
        meta_df = self.create_metadata_dataset()
        
        # Step 4: Validate alignment
        text_aligned, image_aligned, meta_aligned = self.validate_alignment(
            text_df, image_df, meta_df
        )
        
        # Step 5: Generate report
        self.generate_report()
        
        return {
            'text': text_aligned,
            'image': image_aligned,
            'metadata': meta_aligned,
            'stats': self.stats
        }


def main():
    """Main entry point for the preprocessing pipeline."""
    import argparse
    
    # Default paths
    default_input = Path(__file__).parent / 'Training_Data - Train.csv'
    default_output = Path(__file__).parent / 'processed_data'
    
    parser = argparse.ArgumentParser(
        description='E-Commerce Fraud Detection Data Preprocessing Pipeline'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(default_input),
        help='Path to input CSV dataset'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(default_output),
        help='Output directory for processed datasets'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    preprocessor = DataPreprocessor(args.input, args.output)
    results = preprocessor.run()
    
    return results


if __name__ == '__main__':
    main()
