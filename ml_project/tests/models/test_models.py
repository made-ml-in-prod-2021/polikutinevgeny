from pathlib import Path
from typing import List

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

from heart_disease.entities.model_config import TrainModelConfig, ModelType
from heart_disease.models.model import train_model, predict_model, evaluate_model, serialize_model, deserialize_model, \
    save_metrics


def test_train(features: np.ndarray, target: np.ndarray):
    config = TrainModelConfig(model=ModelType.random_forest, random_state=42, params=dict(n_estimators=50))
    model = train_model(features, target, config)
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == features.shape[0]


def test_predict(features: np.ndarray, target: np.ndarray):
    config = TrainModelConfig(model=ModelType.random_forest, random_state=42, params=dict(n_estimators=50))
    model = train_model(features, target, config)
    predicted = predict_model(model, features)
    assert predicted.shape[0] == target.shape[0]


def test_evaluate(features: np.ndarray, target: np.ndarray, metrics: List[str]):
    config = TrainModelConfig(model=ModelType.random_forest, random_state=42, params=dict(n_estimators=50))
    model = train_model(features, target, config)
    metric_values = evaluate_model(model, features, target, metrics)
    assert set(metric_values.keys()) == set(metrics)


def test_serialize(features: np.ndarray, target: np.ndarray, tmpdir: Path):
    path = str(tmpdir / "model.pkl")
    config = TrainModelConfig(model=ModelType.random_forest, random_state=42, params=dict(n_estimators=50))
    model = train_model(features, target, config)
    serialize_model(model, path)
    deserialized = deserialize_model(path)
    assert np.all(model.predict(features) == deserialized.predict(features))


def test_save_metrics(features: np.ndarray, target: np.ndarray, metrics: List[str], tmpdir: Path):
    path = str(tmpdir / "metrics.yaml")
    config = TrainModelConfig(model=ModelType.random_forest, random_state=42, params=dict(n_estimators=50))
    model = train_model(features, target, config)
    metric_values = evaluate_model(model, features, target, metrics)
    save_metrics(metric_values, path)
    with open(path, "r") as f:
        assert metric_values == yaml.safe_load(f)
