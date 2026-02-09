"""
Test the trained spam_sms model with sample messages
"""

import os
import sys
import pickle

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import preprocess

def test_model():
    """Test the trained model with sample messages."""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load the saved model and vectorizer
    model_path = os.path.join(base_dir, 'models', 'spam_classifier_sms.pkl')
    vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer_sms.pkl')
    
    print("Loading trained model and vectorizer...")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("Model loaded successfully!\n")
    
    # Test messages
    test_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at 12?",
        "FREE entry in 2 a weekly competition to win FA Cup final tickets!",
        "Hi mom, I'll be home late tonight. Don't wait for dinner.",
        "URGENT! Your account has been compromised. Click this link immediately to secure it.",
        "Can you pick up some milk on your way home?",
        "Claim your prize now! Call 09061701461 to receive your £900 reward!",
        "Meeting rescheduled to 3 PM tomorrow in conference room B",
        "Win a free iPhone! Text WIN to 87121 now!",
        "Don't forget we have a team meeting at 10 AM on Monday"
    ]
    
    print("="*70)
    print(" "*20 + "MODEL TESTING")
    print("="*70)
    print()
    
    results = []
    spam_count = 0
    ham_count = 0
    
    for i, message in enumerate(test_messages, 1):
        # Preprocess the message
        cleaned_message = preprocess.clean_text(message)
        message_tfidf = vectorizer.transform([cleaned_message])
        prediction = classifier.predict(message_tfidf)[0]
        
        # Get prediction probability
        proba = classifier.predict_proba(message_tfidf)[0]
        confidence = max(proba) * 100
        
        if prediction == 'spam':
            spam_count += 1
            emoji = "[SPAM]"
        else:
            ham_count += 1
            emoji = "[HAM]"
        
        print(f"{i}. Message: {message[:60]}...")
        print(f"   Prediction: {emoji} {prediction.upper()} (Confidence: {confidence:.1f}%)")
        print()
        
        results.append({
            'message': message,
            'prediction': prediction,
            'confidence': confidence
        })
    
    print("="*70)
    print(f"SUMMARY: {spam_count} SPAM, {ham_count} HAM")
    print("="*70)
    
    return results

if __name__ == "__main__":
    # Make sure NLTK data is downloaded
    import preprocess
    preprocess.download_nltk_data()
    
    test_model()
