import pandas as pd
import os

def load_data(file_path):
    """
    Load the SMS Spam Collection dataset.
    
    Args:
        file_path (str): Path to spam.csv file
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    print(f"Loading dataset from {file_path}...")
    
    try:
        # Load the SMS Spam Collection dataset
        # The dataset uses tab separator and has columns: label, text
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Standardize column names
        # SMS Spam Collection typically has columns: v1, v2, v3, v4, v5
        # We only need v1 (label) and v2 (text)
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
        elif 'label' in df.columns and 'text' in df.columns:
            df = df[['label', 'text']]
        else:
            print(f"Error: Expected columns 'v1', 'v2' or 'label', 'text'")
            return None
        
        # Remove any missing values
        df = df.dropna()
        
        print(f"Dataset loaded successfully!")
        print(f"Total messages: {len(df)}")
        print(f"Class distribution:")
        print(df['label'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
