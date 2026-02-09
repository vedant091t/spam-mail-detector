from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    """
    Train a Naive Bayes classifier for spam detection.
    
    Args:
        X_train: Training feature matrix (TF-IDF vectors)
        y_train: Training labels ('spam' or 'ham')
        
    Returns:
        Trained MultinomialNB model
    """
    # Multinomial Naive Bayes is the standard choice for text classification
    # It works well with word count features like TF-IDF
    model = MultinomialNB()
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model

# TODO: Add hyperparameter tuning using GridSearchCV
# TODO: Compare performance with Logistic Regression
