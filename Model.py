from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to train a Random Forest model
def train_model(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X_test, y_test
