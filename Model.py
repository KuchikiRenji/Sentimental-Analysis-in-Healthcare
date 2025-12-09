from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Function to train a Random Forest model
def train_model(X, y):
    """
    Train a Random Forest classifier model.
    
    Args:
        X: Feature matrix (TF-IDF vectors)
        y: Target labels (sentiment categories)
    
    Returns:
        Tuple of (model, X_test, y_test) where model is the trained classifier
    
    Raises:
        ValueError: If input data is invalid or empty
        Exception: If model training fails
    """
    try:
        # Validate input data
        if X is None or y is None:
            raise ValueError("Input data X and y cannot be None")
        
        if X.shape[0] == 0:
            raise ValueError("Feature matrix X is empty")
        
        if len(y) == 0:
            raise ValueError("Target labels y is empty")
        
        if X.shape[0] != len(y):
            raise ValueError(f"Feature matrix X has {X.shape[0]} samples but y has {len(y)} samples")
        
        # Check for valid labels
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Need at least 2 unique classes for classification, found {len(unique_labels)}")
        
        logger.info(f"Training model with {X.shape[0]} samples and {X.shape[1]} features")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Split the dataset into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        logger.info(f"Training set size: {X_train.shape[0]}, Validation set size: {X_test.shape[0]}")
        
        # Train the Random Forest model
        logger.info("Training Random Forest classifier...")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        logger.info("Model training completed successfully")
        
        return model, X_test, y_test
        
    except ValueError as e:
        logger.error(f"Data validation error during model training: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {str(e)}")
        raise
