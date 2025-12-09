from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Function to apply TF-IDF vectorizer on the cleaned reviews
def apply_tfidf(data, vectorizer=None):
    """
    Apply TF-IDF vectorization on the cleaned reviews.
    
    Args:
        data: DataFrame with 'clean_review' column
        vectorizer: Optional pre-fitted TfidfVectorizer. If None, creates and fits a new one.
    
    Returns:
        X: TF-IDF transformed matrix
        tfidf_vectorizer: Fitted TfidfVectorizer
    
    Raises:
        ValueError: If data is invalid or missing required columns
        Exception: If TF-IDF vectorization fails
    """
    try:
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
        
        if 'clean_review' not in data.columns:
            raise ValueError("Input data must contain 'clean_review' column")
        
        # Check for valid reviews
        valid_reviews = data['clean_review'].notna() & (data['clean_review'].str.len() > 0)
        if valid_reviews.sum() == 0:
            raise ValueError("No valid reviews found in 'clean_review' column")
        
        if vectorizer is None:
            logger.info("Creating and fitting new TF-IDF vectorizer...")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            X = tfidf_vectorizer.fit_transform(data['clean_review'])
            logger.info(f"TF-IDF vectorizer fitted. Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
        else:
            logger.info("Using pre-fitted TF-IDF vectorizer...")
            tfidf_vectorizer = vectorizer
            X = tfidf_vectorizer.transform(data['clean_review'])
            logger.info(f"Transformed {X.shape[0]} documents with {X.shape[1]} features")
        
        return X, tfidf_vectorizer
        
    except ValueError as e:
        logger.error(f"Data validation error in TF-IDF: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during TF-IDF vectorization: {str(e)}")
        raise
