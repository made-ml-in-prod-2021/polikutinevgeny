from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

from heart_disease.entities.pipeline_config import PredictConfig
from heart_disease.models.predict_model import predict


def test_train_pipeline(
        train_artifacts: Tuple[str, str, str, str]
):
    metrics_path, model_save_path, pipeline_save_path, metadata_path = train_artifacts
    assert Path(model_save_path).exists()
    assert Path(pipeline_save_path).exists()
    assert Path(metadata_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as f:
        metric_values = yaml.safe_load(f)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0


def test_predict_pipeline(
        tmpdir: Path,
        train_artifacts: Tuple[str, str, str, str],
        test_dataset_file: str,
        dataset_size: int,
):
    metrics_path, model_save_path, pipeline_save_path, metadata_path = train_artifacts
    output = str(tmpdir / "output.txt")
    config = PredictConfig(
        model_load_path=model_save_path,
        pipeline_load_path=pipeline_save_path,
        data_path=test_dataset_file,
        output_path=output
    )
    predict(config)
    assert Path(output).exists()
    with open(output, "r") as f:
        preds = np.array([int(i) for i in f.readlines()])
        assert preds.shape[0] == dataset_size
