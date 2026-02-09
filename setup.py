#!/usr/bin/env python3
"""
Setup script for automated environment configuration
This script automates the installation of dependencies and NLTK data
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print(" "*15 + "SPAM MAIL DETECTOR - SETUP")
    print("="*70)
    
    # Step 1: Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Step 2: Upgrade pip
    print("\n[1/4] Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Pip upgrade")
    
    # Step 3: Install dependencies
    print("\n[2/4] Installing dependencies from requirements.txt...")
    if run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                   "Dependencies installation"):
        print("\n✓ All packages installed successfully!")
    else:
        print("\n✗ Failed to install dependencies. Please check requirements.txt")
        sys.exit(1)
    
    # Step 4: Download NLTK data
    print("\n[3/4] Downloading NLTK data (punkt, stopwords)...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"✗ NLTK download failed: {e}")
        sys.exit(1)
    
    # Step 5: Verify installation
    print("\n[4/4] Verifying installation...")
    try:
        import pandas
        import numpy
        import sklearn
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        print("✓ All packages verified!")
    except ImportError as e:
        print(f"✗ Verification failed: {e}")
        sys.exit(1)
    
    # Step 6: Check if model exists
    print("\n" + "="*70)
    print("  CHECKING MODEL STATUS")
    print("="*70)
    
    model_path = os.path.join('models', 'spam_classifier_sms.pkl')
    if os.path.exists(model_path):
        print(f"\n✓ Pre-trained model found at: {model_path}")
        print("  You can start making predictions immediately!")
        print("\n  Try: python predict.py \"Your message here\"")
    else:
        print(f"\n✗ Pre-trained model not found.")
        print("  Run training first: python src/train_sms.py")
    
    # Summary
    print("\n" + "="*70)
    print("  SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train the model: python src/train_sms.py")
    print("  2. Test the model:  python src/test_sms_model.py")
    print("  3. Make predictions: python predict.py --interactive")
    print("\nFor detailed instructions, see:")
    print("  - QUICK_START.md for quick usage guide")
    print("  - REPRODUCIBILITY.md for training instructions")
    print("  - README.md for complete documentation")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
