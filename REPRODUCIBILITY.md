# Reproducibility Guide

This document provides complete instructions for reproducing the spam detection model training and achieving the same results (97.76% accuracy on the SMS dataset).

---

## Environment Requirements

### Python Version
- **Required**: Python 3.8 or higher
- **Tested on**: Python 3.8, 3.9, 3.10, 3.11
- **Recommended**: Python 3.10

### Operating Systems
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+, Debian, CentOS)
- ✅ macOS 11+

### Hardware Requirements
- **Minimum**: 2GB RAM, 1GB free disk space
- **Recommended**: 4GB+ RAM for comfortable execution
- **Training time**: 5-10 seconds on modern hardware (2020+)

---

## Step 1: Environment Setup

### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n spam-detector python=3.10
conda activate spam-detector
```

---

## Step 2: Install Dependencies

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

### Expected Packages and Versions

The following packages will be installed:

```
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical operations
scikit-learn==1.3.0    # Machine learning algorithms
nltk==3.8.1            # Natural language processing
```

### Verify Installation

```bash
python -c "import pandas, numpy, sklearn, nltk; print('All packages installed successfully!')"
```

**Expected output:** `All packages installed successfully!`

---

## Step 3: Download NLTK Data

The model requires NLTK's punkt tokenizer and stopwords corpus.

### Automatic Download (Recommended)

The training scripts automatically download required NLTK data on first run. No manual action needed.

### Manual Download (if needed)

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Expected output:**
```
[nltk_data] Downloading package punkt to ...
[nltk_data] Downloading package stopwords to ...
```

### Verify NLTK Data

```bash
python -c "from nltk.corpus import stopwords; from nltk.tokenize import word_tokenize; print('NLTK data ready!')"
```

---

## Step 4: Obtain the Dataset

### Option A: Dataset Already Included

The project includes `data/spam_sms.csv` - you can skip this step if the file exists.

### Option B: Download from Source

If the dataset is not included:

1. **Download from UCI ML Repository:**
   - URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
   - File: `SMSSpamCollection` or `spam.csv`

2. **Place in data directory:**
   ```bash
   # Create data directory if it doesn't exist
   mkdir -p data
   
   # Copy downloaded file
   cp /path/to/downloaded/SMSSpamCollection data/spam_sms.csv
   ```

3. **Verify file format:**
   - Columns: `v1` (label), `v2` (text)
   - Encoding: UTF-8 or Latin-1
   - Total rows: 5,572 (including header)

---

## Step 5: Train the Model

### Run Training Script

```bash
python src/train_sms.py
```

### Expected Training Process

The training script will:

1. **Download NLTK data** (if not already present)
   ```
   [1/7] Setting up NLTK resources...
   [nltk_data] punkt is already up-to-date!
   [nltk_data] stopwords is already up-to-date!
   ```

2. **Load dataset** (5,572 messages)
   ```
   [2/7] Loading dataset...
   Dataset loaded successfully!
   Total messages: 5572
   Class distribution:
   ham     4825
   spam     747
   ```

3. **Preprocess text** (~2 seconds)
   ```
   [3/7] Preprocessing text...
   (Lowercasing, removing punctuation, removing stopwords)
   ```

4. **Split data** (80/20 split)
   ```
   [4/7] Splitting data into train and test sets...
   Training set: 4457 messages
   Test set: 1115 messages
   ```

5. **Extract features** (TF-IDF, 3000 features)
   ```
   [5/7] Extracting features using TF-IDF...
   Feature matrix shape: (4457, 3000)
   ```

6. **Train model** (<1 second)
   ```
   [6/7] Training Naive Bayes model...
   Model trained successfully!
   ```

7. **Evaluate model**
   ```
   [7/7] Evaluating model performance...
   
   Accuracy: 0.9776
   Precision: 1.0000
   Recall: 0.8333
   F1-Score: 0.9091
   ```

8. **Save model files**
   ```
   [SAVING] Saving model and vectorizer...
   Model saved to: models/spam_classifier_sms.pkl
   Vectorizer saved to: models/tfidf_vectorizer_sms.pkl
   ```

### Training Duration

- **Total time**: 5-10 seconds on modern hardware
- **CPU usage**: Single-core, minimal
- **Memory usage**: <500MB peak

---

## Step 6: Verify Results

### Expected Performance Metrics

Your trained model should achieve the following results (±0.5% variance due to randomization):

```
Accuracy:  97.76% ± 0.5%
Precision: 100.00% ± 0.5%
Recall:    83.33% ± 2%
F1-Score:  90.91% ± 1%
```

### Check Training Report

```bash
# View detailed results
cat report/training_results_sms.txt

# Or on Windows:
type report\training_results_sms.txt
```

### Verify Model Files

```bash
# Check that model files were created
ls -lh models/spam_classifier_sms.pkl
ls -lh models/tfidf_vectorizer_sms.pkl

# On Windows:
dir models\spam_classifier_sms.pkl
dir models\tfidf_vectorizer_sms.pkl
```

**Expected file sizes:**
- `spam_classifier_sms.pkl`: ~95-100 KB
- `tfidf_vectorizer_sms.pkl`: ~105-110 KB

---

## Step 7: Test the Model

### Run Test Script

```bash
python src/test_sms_model.py
```

**Expected output:** Predictions for 10 test messages with confidence scores

### Run Prediction CLI

```bash
python predict.py "Congratulations! You won a prize!"
```

**Expected output:**
```
Loading spam detection model...
Model loaded successfully!

Message: Congratulations! You won a prize!
Prediction: [SPAM] SPAM (Confidence: 99.XX%)
```

---

## Reproducibility Parameters

All randomness is controlled for reproducibility:

### Random Seeds

| Component | Parameter | Value |
|-----------|-----------|-------|
| Train/Test Split | `random_state` | 42 |
| Model (if applicable) | `random_state` | 42 |

### Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 1.0 | Laplace smoothing (Naive Bayes) |
| `test_size` | 0.2 | 20% of data for testing |
| `max_features` | 3000 | Maximum TF-IDF features |

### Preprocessing Settings

- **Lowercase**: All text converted to lowercase
- **Punctuation**: Removed using `string.punctuation`
- **Numbers**: Removed using regex
- **Stopwords**: NLTK English stopwords
- **Tokenization**: NLTK word_tokenize

---

## Expected Results Summary

After following these steps, you should have:

✅ Trained model with **97.76% accuracy** (±0.5%)  
✅ Model files saved in `models/` directory (~200KB total)  
✅ Training report in `report/training_results_sms.txt`  
✅ Working prediction CLI tool  
✅ All tests passing

---

## Troubleshooting

### Issue: NLTK Data Not Found

**Error:**
```
LookupError: Resource punkt not found
```

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### Issue: CSV Encoding Error

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**Solution:**

The dataset uses Latin-1 encoding. The data loader already handles this with:
```python
pd.read_csv(file_path, encoding='latin-1')
```

If you still encounter issues, try:
```python
pd.read_csv(file_path, encoding='ISO-8859-1')
```

---

### Issue: Different Accuracy Results

**Possible causes:**

1. **Different Python/package versions**
   - Solution: Use exact versions from `requirements.txt`

2. **Different dataset**
   - Solution: Verify dataset has exactly 5,572 rows
   - Check: `wc -l data/spam_sms.csv` should return 5573 (including header)

3. **Random seed not set**
   - Solution: Verify `random_state=42` in train_test_split

4. **NLTK stopwords version**
   - Solution: Re-download NLTK data: `nltk.download('stopwords', force=True)`

**Expected variance:** ±0.5% accuracy due to minor package version differences

---

### Issue: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
pip install scikit-learn==1.3.0
```

---

### Issue: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Close other applications
- Reduce `max_features` from 3000 to 2000 in `feature_extraction.py`
- Use 64-bit Python (not 32-bit)

---

## Verification Checklist

Use this checklist to verify successful reproduction:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed from requirements.txt
- [ ] NLTK data downloaded (punkt, stopwords)
- [ ] Dataset present in `data/spam_sms.csv` (5,572 rows)
- [ ] Training script runs without errors
- [ ] Model achieves 97.76% accuracy (±0.5%)
- [ ] Model files created in `models/` directory
- [ ] Training report generated in `report/` directory
- [ ] Prediction CLI works correctly
- [ ] Test script produces expected results

---

## Contact & Support

If you encounter issues not covered in this guide:

1. **Check existing documentation:**
   - `README.md` - Main project documentation
   - `QUICK_START.md` - Quick usage guide
   - `report/TRAINING_SUMMARY.md` - Detailed performance analysis

2. **Verify environment:**
   ```bash
   python --version
   pip list
   ```

3. **Check file structure:**
   ```bash
   ls -R  # Linux/macOS
   tree   # If tree is installed
   dir /s # Windows
   ```

---

## Citation

If you use this project or reproduce the results, please cite:

**Dataset:**
```
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.
SMS Spam Collection v.1
2011.
Available at: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
```

**Project:**
```
Vedant Tandel
Spam Mail Detector v2.0
2026
AI & ML Internship Project
```

---

**Last Updated:** February 6, 2026  
**Reproducibility Status:** ✅ Fully Reproducible  
**Expected Success Rate:** 99%+ (with proper environment setup)
