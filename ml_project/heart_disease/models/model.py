import pickle
from typing import Union, Dict, List

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import get_scorer

from heart_disease.entities.model_config import TrainModelConfig, ModelType

Classifier = Union[RandomForestClassifier, ExtraTreesClassifier]

_models = {
    ModelType.random_forest: RandomForestClassifier,
    ModelType.extra_trees: ExtraTreesClassifier
}


def train_model(features: np.ndarray, target: np.ndarray, config: TrainModelConfig) -> Classifier:
    model = _models[config.model](random_state=config.random_state, **config.params)
    model.fit(features, target)
    return model


def predict_model(model: Classifier, features: np.ndarray) -> np.ndarray:
    predicted = model.predict(features)
    return predicted


def evaluate_model(model: Classifier, features: np.ndarray, target: np.ndarray, metrics: List[str]) -> Dict[str, float]:
    metric_values = {}
    for metric in metrics:
        scorer = get_scorer(metric)
        metric_values[metric] = scorer(model, features, target).item()
    return metric_values


def save_metrics(metrics: Dict[str, float], path: str):
    with open(path, "w") as f:
        yaml.safe_dump(metrics, f)


def serialize_model(model: Classifier, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def deserialize_model(path: str) -> Classifier:
    with open(path, "rb") as f:
        return pickle.load(f)
