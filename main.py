import pandas as pd
import argparse
import sys
import logging
from preprocessing import load_and_preprocess_data
from sentiment_labeling import apply_sentiment_labeling
from TF_IDF import apply_tfidf
from Model import train_model
from evaluation import evaluate_model
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments for file paths.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Sentimental Analysis in Healthcare - Train and evaluate sentiment classification model'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default='drugsComTrain_raw.tsv',
        help='Path to training data file (default: drugsComTrain_raw.tsv)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='drugsComTest_raw.tsv',
        help='Path to test data file (default: drugsComTest_raw.tsv)'
    )
    parser.add_argument(
        '--skip-interactive',
        action='store_true',
        help='Skip interactive query mode after training'
    )
    return parser.parse_args()


def main_pipeline(train_file, test_file):
    """
    Execute the main sentiment analysis pipeline.
    
    Args:
        train_file: Path to training data file
        test_file: Path to test data file
    
    Returns:
        Tuple of (model, train_data, test_data, tfidf_vectorizer) or None if pipeline fails
    """
    try:
        # Step 1: Load and preprocess the data (separately for train and test)
        logger.info("Step 1: Loading and preprocessing the data...")
        print("Step 1: Loading and preprocessing the data...")
        for _ in tqdm(range(1), desc="Loading & Preprocessing"):
            train_data, test_data = load_and_preprocess_data(train_file, test_file)

        # Step 2: Apply sentiment labeling
        logger.info("Step 2: Applying sentiment labeling...")
        print("\nStep 2: Applying sentiment labeling...")
        for _ in tqdm(range(1), desc="Sentiment Labeling"):
            train_data = apply_sentiment_labeling(train_data)
            test_data = apply_sentiment_labeling(test_data)

        # Step 3: Apply TF-IDF vectorization
        logger.info("Step 3: Applying TF-IDF vectorization...")
        print("\nStep 3: Applying TF-IDF vectorization...")
        for _ in tqdm(range(1), desc="TF-IDF Vectorization"):
            # Fit TF-IDF on training data only
            X_train, tfidf_vectorizer = apply_tfidf(train_data)
            # Transform test data using the same vectorizer
            X_test = tfidf_vectorizer.transform(test_data['clean_review'])

        # Step 4: Train the sentiment classification model
        logger.info("Step 4: Training the sentiment classification model...")
        print("\nStep 4: Training the sentiment classification model...")
        y_train = train_data['sentiment_category']
        for _ in tqdm(range(1), desc="Model Training"):
            model, X_val, y_val = train_model(X_train, y_train)

        # Step 5: Evaluate the model on the separate test set
        logger.info("Step 5: Evaluating the model on test data...")
        print("\nStep 5: Evaluating the model on test data...")
        y_test = test_data['sentiment_category']
        for _ in tqdm(range(1), desc="Model Evaluation"):
            evaluate_model(model, X_test, y_test)
        
        return model, train_data, test_data, tfidf_vectorizer
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"ERROR: {str(e)}")
        print("Please ensure the data files exist and the paths are correct.")
        return None
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        print(f"ERROR: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        return None

# Sentiment categorization function
def categorize_sentiment(score):
    """
    Convert continuous sentiment score to discrete category.
    
    Args:
        score: Continuous sentiment score
    
    Returns:
        Integer: 1 for positive, -1 for negative, 0 for neutral
    """
    if pd.isna(score):
        return 0
    if score > 0:
        return 1  # Positive sentiment
    elif score < 0:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Main function for user interaction
def interactive_query(train_data, test_data):
    """
    Interactive mode for querying sentiment by medication and condition.
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
    """
    try:
        # Combine train and test data for querying
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # User input for medication
        medication = input("\nEnter the drug name you are using (e.g., Mirtazapine) or 'quit' to exit: ").strip().lower()
        
        if medication == 'quit':
            print("Exiting interactive mode.")
            return

        # Optional input for condition
        condition = input("Enter your condition (optional) (e.g., Depression): ").strip().lower()

        # Filter reviews based on medication and optionally the condition
        if condition:
            medication_reviews = all_data[(all_data['drugName'].str.lower() == medication) &
                                          (all_data['condition'].str.lower() == condition)]
        else:
            medication_reviews = all_data[all_data['drugName'].str.lower() == medication]

        if not medication_reviews.empty:
            sentiment = medication_reviews['sentiment_category'].mean()
            sentiment_category = categorize_sentiment(sentiment)
            sentiment_name = {1: "Positive", -1: "Negative", 0: "Neutral"}.get(sentiment_category, "Unknown")
            print(f"\nSentiment for {medication}" + (f" (condition: {condition})" if condition else "") + f": {sentiment_name} ({sentiment_category})")
            print(f"Found {len(medication_reviews)} reviews")
        else:
            print("No reviews found for the specified medication or condition.")
            
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode.")
    except Exception as e:
        logger.error(f"Error in interactive query: {str(e)}")
        print(f"ERROR: An error occurred during query: {str(e)}")


def main():
    """
    Main entry point for the sentiment analysis application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info(f"Starting sentiment analysis pipeline with train_file={args.train_file}, test_file={args.test_file}")
    
    # Run the main pipeline
    result = main_pipeline(args.train_file, args.test_file)
    
    if result is None:
        logger.error("Pipeline failed. Exiting.")
        sys.exit(1)
    
    model, train_data, test_data, tfidf_vectorizer = result
    
    # Interactive query mode (unless skipped)
    if not args.skip_interactive:
        try:
            interactive_query(train_data, test_data)
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"ERROR: An error occurred in interactive mode: {str(e)}")
    
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
