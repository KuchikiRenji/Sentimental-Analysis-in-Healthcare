from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import pandas as pd
from utils import ensure_nltk_resources

# Configure logging
logger = logging.getLogger(__name__)

# Ensure VADER lexicon is downloaded
try:
    ensure_nltk_resources(['vader_lexicon'])
except Exception as e:
    logger.error(f"Failed to download VADER lexicon: {str(e)}")
    raise

# Initialize VADER sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize VADER sentiment analyzer: {str(e)}")
    raise

# Convert continuous sentiment scores into discrete classes (positive, negative, neutral)
def categorize_sentiment(score):
    """
    Convert continuous sentiment score to discrete category.
    
    Args:
        score: Continuous sentiment score (typically -1 to 1)
    
    Returns:
        Integer: 1 for positive, -1 for negative, 0 for neutral
    """
    try:
        if pd.isna(score):
            return 0
        if score > 0:
            return 1  # Positive sentiment
        elif score < 0:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment
    except Exception as e:
        logger.warning(f"Error categorizing sentiment score {score}: {str(e)}")
        return 0

# Label the sentiment based on VADER scores
def apply_sentiment_labeling(data):
    """
    Apply sentiment labeling to data using VADER sentiment analyzer.
    
    Args:
        data: DataFrame with 'review' column
    
    Returns:
        DataFrame with added 'sentiment' and 'sentiment_category' columns
    
    Raises:
        ValueError: If data is invalid or missing required columns
        Exception: If sentiment analysis fails
    """
    try:
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
        
        if 'review' not in data.columns:
            raise ValueError("Input data must contain 'review' column")
        
        logger.info(f"Applying sentiment labeling to {len(data)} reviews...")
        
        def get_sentiment_score(review):
            try:
                if pd.isna(review) or not isinstance(review, str):
                    return 0.0
                return sia.polarity_scores(review)['compound']
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for review: {str(e)}")
                return 0.0
        
        data = data.copy()
        data['sentiment'] = data['review'].apply(get_sentiment_score)
        data['sentiment_category'] = data['sentiment'].apply(categorize_sentiment)
        
        # Log sentiment distribution
        sentiment_counts = data['sentiment_category'].value_counts().to_dict()
        logger.info(f"Sentiment distribution: {sentiment_counts}")
        
        return data
        
    except ValueError as e:
        logger.error(f"Data validation error in sentiment labeling: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during sentiment labeling: {str(e)}")
        raise
