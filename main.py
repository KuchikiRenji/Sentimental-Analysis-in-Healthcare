import pandas as pd
from preprocessing import load_and_preprocess_data
from sentiment_labeling import apply_sentiment_labeling
from TF_IDF import apply_tfidf
from Model import train_model
from evaluation import evaluate_model
from tqdm import tqdm

# File paths for the train and test dataset
train_file = "drugsComTrain_raw.tsv"
test_file = "drugsComTest_raw.tsv"

# Step 1: Load and preprocess the data (separately for train and test)
print("Step 1: Loading and preprocessing the data...")
for _ in tqdm(range(1), desc="Loading & Preprocessing"):
    train_data, test_data = load_and_preprocess_data(train_file, test_file)

# Step 2: Apply sentiment labeling
print("\nStep 2: Applying sentiment labeling...")
for _ in tqdm(range(1), desc="Sentiment Labeling"):
    train_data = apply_sentiment_labeling(train_data)
    test_data = apply_sentiment_labeling(test_data)

# Step 3: Apply TF-IDF vectorization
print("\nStep 3: Applying TF-IDF vectorization...")
for _ in tqdm(range(1), desc="TF-IDF Vectorization"):
    # Fit TF-IDF on training data only
    X_train, tfidf_vectorizer = apply_tfidf(train_data)
    # Transform test data using the same vectorizer
    X_test = tfidf_vectorizer.transform(test_data['clean_review'])

# Step 4: Train the sentiment classification model
print("\nStep 4: Training the sentiment classification model...")
y_train = train_data['sentiment_category']
for _ in tqdm(range(1), desc="Model Training"):
    model, X_val, y_val = train_model(X_train, y_train)

# Step 5: Evaluate the model on the separate test set
print("\nStep 5: Evaluating the model on test data...")
y_test = test_data['sentiment_category']
for _ in tqdm(range(1), desc="Model Evaluation"):
    evaluate_model(model, X_test, y_test)

# Sentiment categorization function
def categorize_sentiment(score):
    if score > 0:
        return 1  # Positive sentiment
    elif score < 0:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Main function for user interaction
def main():
    # Combine train and test data for querying
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # User input for medication
    medication = input("Enter the drug name you are using (e.g., Mirtazapine): ").strip().lower()

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
        print(f"Sentiment for {medication}: {sentiment_category}")
    else:
        print("No reviews found for the specified medication or condition.")

if __name__ == "__main__":
    main()
