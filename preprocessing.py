import pandas as pd
import re
import nltk
import os
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import ensure_nltk_resources

# Configure logging
logger = logging.getLogger(__name__)

# Ensure required NLTK resources are downloaded
try:
    ensure_nltk_resources(['punkt', 'stopwords', 'vader_lexicon'])
except Exception as e:
    logger.error(f"Failed to download required NLTK resources: {str(e)}")
    raise

stop_words = set(stopwords.words('english'))

# Function to clean the text: remove punctuation, lowercase, tokenize, and remove stopwords
def clean_text(text):
    """
    Clean text by removing punctuation, lowercasing, tokenizing, and removing stopwords.
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = text.lower()  # convert to lowercase
        tokens = word_tokenize(text)  # tokenize text
        filtered_tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
        return ' '.join(filtered_tokens)
    except Exception as e:
        logger.warning(f"Error cleaning text: {str(e)}")
        return ""

# Load and preprocess data
def load_and_preprocess_data(train_file, test_file):
    """
    Load and preprocess training and test datasets.
    
    Args:
        train_file: Path to training data file
        test_file: Path to test data file
    
    Returns:
        Tuple of (trainDataset, testDataset) DataFrames
    
    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If data files are empty or malformed
        Exception: For other data processing errors
    """
    # Validate file existence
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    try:
        # Load the datasets separately
        logger.info(f"Loading training data from {train_file}...")
        trainDataset = pd.read_csv(train_file, sep='\t')
        logger.info(f"Loading test data from {test_file}...")
        testDataset = pd.read_csv(test_file, sep='\t')
        
        # Validate datasets are not empty
        if trainDataset.empty:
            raise ValueError(f"Training dataset is empty: {train_file}")
        if testDataset.empty:
            raise ValueError(f"Test dataset is empty: {test_file}")
        
        # Rename columns for better understanding
        expected_columns = 7
        if len(trainDataset.columns) != expected_columns:
            raise ValueError(f"Training dataset has {len(trainDataset.columns)} columns, expected {expected_columns}")
        if len(testDataset.columns) != expected_columns:
            raise ValueError(f"Test dataset has {len(testDataset.columns)} columns, expected {expected_columns}")
        
        trainDataset.columns = ['Id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
        testDataset.columns = ['Id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
        
        # Log initial data info
        logger.info(f"Training dataset shape before preprocessing: {trainDataset.shape}")
        logger.info(f"Test dataset shape before preprocessing: {testDataset.shape}")
        
        # Drop missing values
        train_initial_count = len(trainDataset)
        test_initial_count = len(testDataset)
        trainDataset.dropna(inplace=True)
        testDataset.dropna(inplace=True)
        
        train_dropped = train_initial_count - len(trainDataset)
        test_dropped = test_initial_count - len(testDataset)
        
        if train_dropped > 0:
            logger.warning(f"Dropped {train_dropped} rows with missing values from training data")
        if test_dropped > 0:
            logger.warning(f"Dropped {test_dropped} rows with missing values from test data")
        
        # Validate datasets are not empty after dropping missing values
        if trainDataset.empty:
            raise ValueError("Training dataset is empty after dropping missing values")
        if testDataset.empty:
            raise ValueError("Test dataset is empty after dropping missing values")
        
        # Apply text cleaning
        logger.info("Applying text cleaning...")
        trainDataset['clean_review'] = trainDataset['review'].apply(clean_text)
        testDataset['clean_review'] = testDataset['review'].apply(clean_text)
        
        # Validate that we have valid reviews after cleaning
        train_valid = trainDataset['clean_review'].str.len() > 0
        test_valid = testDataset['clean_review'].str.len() > 0
        
        if train_valid.sum() == 0:
            raise ValueError("No valid reviews after text cleaning in training data")
        if test_valid.sum() == 0:
            raise ValueError("No valid reviews after text cleaning in test data")
        
        logger.info(f"Training dataset shape after preprocessing: {trainDataset.shape}")
        logger.info(f"Test dataset shape after preprocessing: {testDataset.shape}")
        
        return trainDataset, testDataset
        
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty or malformed: {str(e)}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data loading/preprocessing: {str(e)}")
        raise