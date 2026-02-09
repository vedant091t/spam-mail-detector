import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_data():
    """
    Download necessary NLTK data (stopwords, punkt).
    Quietly handles the download to avoid cluttering output.
    """
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("Done.")

def clean_text(text):
    """
    Preprocess the input text:
    1. Lowercase
    2. Remove punctuation and numbers
    3. Tokenize
    4. Remove stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing punctuation and numbers
    # We replace them with space to avoid merging words (e.g., "hello,world" -> "hello world")
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Join back to string
    return " ".join(filtered_tokens)

if __name__ == "__main__":
    download_nltk_data()
    sample = "Hello! Win a FREE prize calling number 12345."
    print(f"Original: {sample}")
    print(f"Cleaned: {clean_text(sample)}")
