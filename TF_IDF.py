from sklearn.feature_extraction.text import TfidfVectorizer

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
    """
    if vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        X = tfidf_vectorizer.fit_transform(data['clean_review'])
    else:
        tfidf_vectorizer = vectorizer
        X = tfidf_vectorizer.transform(data['clean_review'])
    return X, tfidf_vectorizer
