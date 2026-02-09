"""
Spam Mail Detector - Main Training Pipeline

This script implements a complete ML pipeline for spam detection:
1. Load SMS Spam Collection dataset
2. Preprocess text (clean, tokenize, remove stopwords)
3. Extract features using TF-IDF
4. Train Naive Bayes classifier
5. Evaluate performance
6. Interactive testing mode

Author: Vedant Tandel
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
import preprocess
import feature_extraction
import model
import evaluate
import data_loader

def main():
    """Main execution function for the spam detector."""
    
    print("="*70)
    print(" "*20 + "SPAM MAIL DETECTOR")
    print("="*70)
    
    # Step 1: Download NLTK resources (required for preprocessing)
    print("\n[1/7] Setting up NLTK resources...")
    preprocess.download_nltk_data()
    
    # Step 2: Load the dataset
    print("\n[2/7] Loading dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'spam_sms.csv')
    
    df = data_loader.load_data(data_path)
    
    if df is None:
        print("Error: Failed to load dataset. Exiting...")
        return
    
    # Step 3: Preprocess the text data
    print("\n[3/7] Preprocessing text...")
    print("(Lowercasing, removing punctuation, removing stopwords)")
    df['cleaned_text'] = df['text'].apply(preprocess.clean_text)
    
    # Step 4: Split data into training and testing sets (80/20 split)
    print("\n[4/7] Splitting data into train and test sets...")
    X = df['cleaned_text']  # Features (cleaned text)
    y = df['label']         # Labels (spam/ham)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42     # For reproducibility
    )
    
    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")
    
    # Step 5: Feature extraction using TF-IDF
    print("\n[5/7] Extracting features using TF-IDF...")
    vectorizer = feature_extraction.create_vectorizer(max_features=3000)
    
    # Fit vectorizer on training data and transform both sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Step 6: Train the Naive Bayes model
    print("\n[6/7] Training Naive Bayes model...")
    classifier = model.train_model(X_train_tfidf, y_train)
    
    # Step 7: Evaluate the model on test set
    print("\n[7/7] Evaluating model performance...")
    y_pred = classifier.predict(X_test_tfidf)
    evaluate.evaluate_model(y_test, y_pred)
    
    # Interactive testing mode
    print("="*70)
    print("INTERACTIVE TESTING MODE")
    print("="*70)
    print("Enter a message to classify it as SPAM or HAM.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            user_message = input("Enter message: ")
            
            if user_message.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if not user_message.strip():
                print("Please enter a valid message.\n")
                continue
            
            # Preprocess and predict
            cleaned_message = preprocess.clean_text(user_message)
            message_tfidf = vectorizer.transform([cleaned_message])
            prediction = classifier.predict(message_tfidf)[0]
            
            # Display result
            result = prediction.upper()
            if prediction == 'spam':
                print(f"🚨 Prediction: {result}\n")
            else:
                print(f"✅ Prediction: {result}\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
