
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'

# src/infer.py
import pandas as pd

FEATURE_COLUMNS = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

def predict_json(payload: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    y = int(model.predict(X)[0])
    try:
        conf = float(model.predict_proba(X)[0][1])
    except Exception:
        conf = 0.0
    return {'prediction': y, 'confidence': conf}



def predict_file(input_csv: Path) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv)
    preds = model.predict(df)
    return pd.DataFrame({'prediction': preds})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='JSON string payload for single prediction')
    parser.add_argument('--csv', type=str, help='CSV file path for batch predictions')
    args = parser.parse_args()
    if args.json:
        payload = json.loads(args.json)
        print(json.dumps(predict_json(payload), indent=2))
    elif args.csv:
        out = predict_file(Path(args.csv))
        print(out.head())
    else:
        print('Provide --json or --csv')
