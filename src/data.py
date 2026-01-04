
import os
import pandas as pd
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DATA_PATH = DATA_DIR / 'heart.csv'


def download_uci_heart(force: bool = False) -> Path:
    """Download the Cleveland heart dataset to data/heart.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    try:
        import requests
        r = requests.get(UCI_URL, timeout=30)
        r.raise_for_status()
        DATA_PATH.write_text(r.text)
        return DATA_PATH
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")


def load_dataframe() -> pd.DataFrame:
    """Load dataset into a DataFrame with proper column names and basic cleaning."""
    if not DATA_PATH.exists():
        download_uci_heart()
    df = pd.read_csv(DATA_PATH, names=COLUMNS)
    # Replace '?' with NA and coerce types
    df = df.replace('?', pd.NA)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Binary target: 0=healthy, >0=disease
    df['target'] = (df['target'] > 0).astype(int)
    return df

if __name__ == '__main__':
    p = download_uci_heart()
    print(f"Downloaded to {p}")
