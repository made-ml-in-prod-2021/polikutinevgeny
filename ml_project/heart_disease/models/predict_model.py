import logging
from pathlib import Path

import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from heart_disease.data.make_dataset import read_data
from heart_disease.entities.pipeline_config import PredictConfig
from heart_disease.features.build_features import deserialize_pipeline
from heart_disease.models.model import deserialize_model, predict_model

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

cs = ConfigStore.instance()
cs.store(name="predict_config", node=PredictConfig)


def save_predictions(predictions: np.ndarray, path: str):
    with open(path, "w") as f:
        print(*predictions, sep="\n", file=f)


@hydra.main(config_path=PROJECT_ROOT / "config", config_name="predict_config")
def predict(cfg: PredictConfig):
    data = read_data(to_absolute_path(cfg.data_path))
    pipeline = deserialize_pipeline(to_absolute_path(cfg.pipeline_load_path))
    model = deserialize_model(to_absolute_path(cfg.model_load_path))
    predicted = predict_model(model, pipeline.transform(data))
    save_predictions(predicted, to_absolute_path(cfg.output_path))


if __name__ == '__main__':
    predict()
