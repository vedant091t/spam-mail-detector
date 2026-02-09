from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer(max_features=3000):
    """
    Create a TF-IDF vectorizer for converting text to numerical features.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text documents 
    into numerical vectors based on word importance.
    
    Args:
        max_features (int): Maximum number of features (words) to keep
        
    Returns:
        TfidfVectorizer object
    """
    # Create TF-IDF vectorizer with standard parameters
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    return vectorizer

# TODO: Experiment with n-grams (bigrams, trigrams)
# TODO: Try different max_features values (1000, 5000, etc.)
