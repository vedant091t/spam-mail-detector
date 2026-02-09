"""
Spam Mail Detector - Training Script for spam_sms.csv
This script trains the model and saves the results to a file.
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import pickle

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
import preprocess
import feature_extraction
import model
import evaluate
import data_loader

def train_and_evaluate():
    """Train model on spam_sms.csv and save results."""
    
    results = []
    results.append("="*70)
    results.append(" "*20 + "SPAM MAIL DETECTOR - SMS DATASET")
    results.append("="*70)
    
    # Step 1: Download NLTK resources
    print("\n[1/7] Setting up NLTK resources...")
    results.append("\n[1/7] Setting up NLTK resources...")
    preprocess.download_nltk_data()
    
    # Step 2: Load the dataset
    print("\n[2/7] Loading dataset...")
    results.append("\n[2/7] Loading dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'spam_sms.csv')
    
    df = data_loader.load_data(data_path)
    
    if df is None:
        print("Error: Failed to load dataset. Exiting...")
        return
    
    results.append(f"Dataset loaded successfully!")
    results.append(f"Total messages: {len(df)}")
    results.append(f"Class distribution:")
    results.append(str(df['label'].value_counts()))
    
    # Step 3: Preprocess the text data
    print("\n[3/7] Preprocessing text...")
    results.append("\n[3/7] Preprocessing text...")
    results.append("(Lowercasing, removing punctuation, removing stopwords)")
    df['cleaned_text'] = df['text'].apply(preprocess.clean_text)
    
    # Step 4: Split data into training and testing sets
    print("\n[4/7] Splitting data into train and test sets...")
    results.append("\n[4/7] Splitting data into train and test sets...")
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42
    )
    
    results.append(f"Training set: {len(X_train)} messages")
    results.append(f"Test set: {len(X_test)} messages")
    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")
    
    # Step 5: Feature extraction using TF-IDF
    print("\n[5/7] Extracting features using TF-IDF...")
    results.append("\n[5/7] Extracting features using TF-IDF...")
    vectorizer = feature_extraction.create_vectorizer(max_features=3000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    results.append(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Step 6: Train the Naive Bayes model
    print("\n[6/7] Training Naive Bayes model...")
    results.append("\n[6/7] Training Naive Bayes model...")
    classifier = model.train_model(X_train_tfidf, y_train)
    
    # Step 7: Evaluate the model
    print("\n[7/7] Evaluating model performance...")
    results.append("\n[7/7] Evaluating model performance...")
    y_pred = classifier.predict(X_test_tfidf)
    
    # Capture evaluation results
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    results.append("\n" + "="*70)
    results.append("MODEL EVALUATION RESULTS")
    results.append("="*70)
    results.append(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    results.append(f"Precision: {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
    results.append(f"Recall: {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
    results.append(f"F1-Score: {f1_score(y_test, y_pred, pos_label='spam'):.4f}")
    results.append("\nClassification Report:")
    results.append(classification_report(y_test, y_pred))
    
    # Print to console
    evaluate.evaluate_model(y_test, y_pred)
    
    # Save model and vectorizer
    print("\n[SAVING] Saving model and vectorizer...")
    results.append("\n[SAVING] Saving model and vectorizer...")
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(models_dir, 'spam_classifier_sms.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    results.append(f"Model saved to: {model_path}")
    print(f"Model saved to: {model_path}")
    
    # Save the vectorizer
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer_sms.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    results.append(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save results to file
    report_dir = os.path.join(base_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'training_results_sms.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    
    print(f"\nTraining results saved to: {report_path}")
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    train_and_evaluate()
