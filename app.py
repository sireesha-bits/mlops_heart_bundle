
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'

app = FastAPI(title='Heart Disease Risk API', version='1.0')
model = None

class Input(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Prometheus metrics
req_counter = Counter('api_requests_total','Total API requests',['endpoint'])
latency = Histogram('api_request_latency_seconds','Request latency',['endpoint'])

@app.on_event('startup')
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError('Model artifact not found. Train first.')

@app.get('/health')
def health():
    return {'status':'ok'}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/predict')
async def predict(payload: Input):
    req_counter.labels(endpoint='/predict').inc()
    with latency.labels(endpoint='/predict').time():
        X = np.array([[payload.age,payload.sex,payload.cp,payload.trestbps,payload.chol,payload.fbs,payload.restecg,payload.thalach,payload.exang,payload.oldpeak,payload.slope,payload.ca,payload.thal]])
        y = int(model.predict(X)[0])
        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception:
            prob = 0.0
        return {'prediction': y, 'confidence': prob}
