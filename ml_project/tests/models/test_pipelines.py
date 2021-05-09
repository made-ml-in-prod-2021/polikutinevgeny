from pathlib import Path
from typing import List, Callable, OrderedDict

import numpy as np
import yaml

from heart_disease.entities.data_loading_config import DataLoadingConfig
from heart_disease.entities.model_config import TrainModelConfig, EvaluateModelConfig, ModelType
from heart_disease.entities.pipeline_config import TrainingConfig, PredictConfig
from heart_disease.entities.splitting_config import SplittingConfig
from heart_disease.models.predict_model import predict
from heart_disease.models.train_model import train_pipeline
from tests.conftest import get_feature_config


def train_model(
        categorical_features: List[str],
        dataset_file: str,
        metrics: List[str],
        numerical_features: List[str],
        statistics: OrderedDict[str, Callable],
        target_column: str,
        tmpdir: Path
):
    model_save_path = str(tmpdir / "model.pkl")
    pipeline_save_path = str(tmpdir / "pipeline.pkl")
    metrics_path = str(tmpdir / "metrics.yaml")
    metadata_path = str(tmpdir / "metadata.pkl")
    projection_features = 5
    polynomial_degree = 2
    n_clusters = 2
    feature_config = get_feature_config(target_column, categorical_features, n_clusters, numerical_features,
                                        polynomial_degree,
                                        projection_features, statistics)
    config = TrainingConfig(
        data_load_config=DataLoadingConfig(
            split_config=SplittingConfig(
                random_state=42,
                val_size=0.2
            ),
            data_path=dataset_file
        ),
        feature_config=feature_config,
        model_config=TrainModelConfig(
            model=ModelType.random_forest,
            random_state=42,
            params=dict(n_estimators=55)
        ),
        evaluation_config=EvaluateModelConfig(
            metrics=metrics,
            metric_file_path=metrics_path
        ),
        pipeline_save_path=pipeline_save_path,
        model_save_path=model_save_path,
        metadata_save_path=metadata_path
    )
    train_pipeline(config)
    return metrics_path, model_save_path, pipeline_save_path, metadata_path


def test_train_pipeline(
        tmpdir: Path,
        dataset_file: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_column: str,
        metrics: List[str],
        statistics: OrderedDict[str, Callable]
):
    metrics_path, model_save_path, pipeline_save_path, metadata_path = train_model(
        categorical_features,
        dataset_file,
        metrics,
        numerical_features,
        statistics,
        target_column,
        tmpdir
    )
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
        dataset_file: str,
        dataset_size: int,
        categorical_features: List[str],
        numerical_features: List[str],
        target_column: str,
        metrics: List[str],
        statistics: OrderedDict[str, Callable]
):
    output = str(tmpdir / "output.txt")
    _, model_save_path, pipeline_save_path, _ = train_model(
        categorical_features,
        dataset_file,
        metrics,
        numerical_features,
        statistics,
        target_column,
        tmpdir
    )
    config = PredictConfig(
        model_load_path=model_save_path,
        pipeline_load_path=pipeline_save_path,
        data_path=dataset_file,
        output_path=output
    )
    predict(config)
    assert Path(output).exists()
    with open(output, "r") as f:
        preds = np.array([int(i) for i in f.readlines()])
        assert preds.shape[0] == dataset_size
