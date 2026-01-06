
# tests/test_inference_dataframe.py
import joblib
import pandas as pd
from pathlib import Path

FEATURE_COLUMNS = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

def test_model_accepts_dataframe():
    model_path = Path('artifacts') / 'model.joblib'
    assert model_path.exists(), "Train first to create model.joblib"

    model = joblib.load(model_path)
    sample = { 'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
               'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
               'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1 }
    X = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
    pred = model.predict(X)
    assert pred.shape == (1,)
