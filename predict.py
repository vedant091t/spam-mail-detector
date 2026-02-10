#!/usr/bin/env python3
"""
Spam Message Predictor - CLI Tool

A standalone command-line interface for spam detection using the trained model.
This tool uses the best-performing model (spam_classifier_sms.pkl) trained on
the SMS Spam Collection dataset with 97.76% accuracy.

Usage:
    python predict.py "Your message here"                 # Single prediction
    python predict.py --interactive                       # Interactive mode
    python predict.py --file messages.txt                 # Batch from file
    python predict.py --json "message"                    # JSON output
    python predict.py --help                              # Show help

Author: Vedant Tandel
"""

import os
import sys
import pickle
import argparse
import json as json_module
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from preprocess import clean_text, download_nltk_data
except ImportError:
    print("Error: Could not import preprocessing module.")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


class SpamPredictor:
    """Spam message predictor using pre-trained model."""
    
    def __init__(self, model_dir='models'):
        """
        Initialize the spam predictor.
        
        Args:
            model_dir (str): Directory containing the trained model files
        """
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer from disk."""
        model_path = self.model_dir / 'spam_classifier_sms.pkl'
        vectorizer_path = self.model_dir / 'tfidf_vectorizer_sms.pkl'
        
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            print("\nPlease train the model first by running:")
            print("  python src/train_sms.py")
            sys.exit(1)
        
        if not vectorizer_path.exists():
            print(f"Error: Vectorizer file not found at {vectorizer_path}")
            print("\nPlease train the model first by running:")
            print("  python src/train_sms.py")
            sys.exit(1)
        
        try:
            print("Loading spam detection model...")
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model loaded successfully!\n")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("The model files might be corrupted. Try retraining.")
            sys.exit(1)
    
    def predict(self, message):
        """
        Predict whether a message is spam or ham.
        
        Args:
            message (str): The message to classify
            
        Returns:
            tuple: (prediction, confidence) where prediction is 'spam' or 'ham'
                   and confidence is a float between 0 and 1
        """
        if not message or not message.strip():
            return None, 0.0
        
        try:
            cleaned_message = clean_text(message)
            message_tfidf = self.vectorizer.transform([cleaned_message])
            prediction = self.classifier.predict(message_tfidf)[0]
            probabilities = self.classifier.predict_proba(message_tfidf)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            return None, 0.0
    
    def predict_and_display(self, message, show_message=True, json_output=False):
        """
        Predict and display result in a user-friendly format.
        
        Args:
            message (str): The message to classify
            show_message (bool): Whether to display the input message
            json_output (bool): Output in JSON format
        """
        prediction, confidence = self.predict(message)
        
        if prediction is None:
            if json_output:
                print(json_module.dumps({"error": "Empty or invalid message"}))
            else:
                print("Error: Empty message\n")
            return
        
        confidence_pct = confidence * 100
        
        if json_output:
            result = {
                "message": message[:100],
                "prediction": prediction,
                "confidence": round(confidence_pct, 2),
                "is_spam": prediction == 'spam'
            }
            print(json_module.dumps(result))
            return
        
        if show_message:
            print(f"Message: {message[:80]}{'...' if len(message) > 80 else ''}")
        
        result = prediction.upper()
        
        if prediction == 'spam':
            status = "[SPAM]"
            emoji = "🚨" if sys.platform != 'win32' else "⚠"
        else:
            status = "[HAM]"
            emoji = "✓" if sys.platform != 'win32' else "✓"
        
        print(f"Prediction: {status} {result} (Confidence: {confidence_pct:.2f}%)")
        print()


def predict_single(predictor, message, json_output=False):
    """Predict a single message."""
    predictor.predict_and_display(message, json_output=json_output)


def predict_interactive(predictor):
    """Interactive prediction mode."""
    print("="*70)
    print(" "*20 + "INTERACTIVE SPAM DETECTOR")
    print("="*70)
    print("Enter messages to classify them as SPAM or HAM.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")
    
    while True:
        try:
            message = input("Enter message: ").strip()
            
            if message.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if not message:
                print("Please enter a valid message.\n")
                continue
            
            print()
            predictor.predict_and_display(message, show_message=False)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def predict_batch(predictor, file_path, json_output=False):
    """Predict messages from a file (one per line)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                messages = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    if json_output:
        results = []
        for message in messages:
            prediction, confidence = predictor.predict(message)
            if prediction:
                results.append({
                    "message": message[:100],
                    "prediction": prediction,
                    "confidence": round(confidence * 100, 2),
                    "is_spam": prediction == 'spam'
                })
        print(json_module.dumps(results, indent=2))
        return
    
    print(f"Processing messages from: {file_path}\n")
    print("="*70)
    
    spam_count = 0
    ham_count = 0
    
    for i, message in enumerate(messages, 1):
        print(f"\n[{i}] ", end="")
        prediction, confidence = predictor.predict(message)
        
        if prediction == 'spam':
            spam_count += 1
        else:
            ham_count += 1
        
        predictor.predict_and_display(message)
    
    print("="*70)
    print(f"SUMMARY: {spam_count} SPAM, {ham_count} HAM")
    print("="*70)


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description='Spam Message Predictor - Classify messages as spam or ham',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "Congratulations! You won a prize!"
  python predict.py --interactive
  python predict.py --file messages.txt
  python predict.py --json "Check this message"

The tool uses a Naive Bayes classifier trained on SMS spam data
with 97.76% accuracy, 100% precision, and 83.33% recall.
        """
    )
    
    parser.add_argument(
        'message',
        nargs='?',
        help='Message to classify (if not using --interactive or --file)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Enter interactive mode for multiple predictions'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Path to file containing messages (one per line)'
    )
    
    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '-m', '--model-dir',
        type=str,
        default='models',
        help='Directory containing model files (default: models)'
    )
    
    args = parser.parse_args()
    
    download_nltk_data()
    
    predictor = SpamPredictor(model_dir=args.model_dir)
    
    if args.interactive:
        if args.json:
            print("Warning: JSON output not supported in interactive mode")
        predict_interactive(predictor)
    elif args.file:
        predict_batch(predictor, args.file, json_output=args.json)
    elif args.message:
        predict_single(predictor, args.message, json_output=args.json)
    else:
        parser.print_help()
        print("\nError: Please provide a message, use --interactive, or specify --file")
        sys.exit(1)


if __name__ == "__main__":
    main()
