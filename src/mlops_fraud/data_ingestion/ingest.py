# src/data_ingestion/ingest.py

import os

import pandas as pd

RAW_DATA_PATH = "data/raw/paysim.csv"
PROCESSED_DATA_PATH = "data/processed/paysim_clean.csv"


def load_raw_data(path):
    """Load raw data from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    print(f"Loading raw data from {path}...")
    return pd.read_csv(path)


def basic_cleaning(df):
    """Perform basic cleaning on the DataFrame."""
    drop_cols = ["oldbalanceOrg", "newbalanceOrg", "oldbalanceDest", "newbalanceDest"]
    df_cleaned = df.drop(columns=drop_cols, errors="ignore")
    print(f"Columns after dropping: {df_cleaned.columns.tolist()}")
    return df_cleaned


def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    # Load raw data
    df_raw = load_raw_data(RAW_DATA_PATH)
    df_clean = basic_cleaning(df_raw)
    save_cleaned_data(df_clean, PROCESSED_DATA_PATH)
