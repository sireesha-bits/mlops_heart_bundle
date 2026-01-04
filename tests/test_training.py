
from src.train import train

def test_train_runs():
    metrics = train(model_name='lr', mlflow_enabled=False)
    assert 'accuracy' in metrics and metrics['accuracy'] >= 0.5
