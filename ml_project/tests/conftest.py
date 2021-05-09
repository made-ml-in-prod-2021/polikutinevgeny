from collections import OrderedDict
from typing import List, Union, Dict, Callable, Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.random import Generator, PCG64

from heart_disease.data.make_dataset import read_data
from heart_disease.entities.data_loading_config import DataLoadingConfig
from heart_disease.entities.feature_config import FeatureConfig, RandomProjectionFeaturesConfig, \
    StatisticalFeaturesConfig, KMeansFeaturesConfig, PolynomialFeaturesConfig, RawFeaturesConfig
from heart_disease.entities.model_config import TrainModelConfig, EvaluateModelConfig, ModelType
from heart_disease.entities.pipeline_config import TrainingConfig
from heart_disease.entities.splitting_config import SplittingConfig
from heart_disease.features.build_features import build_feature_pipeline, extract_raw_features
from heart_disease.models.train_model import train_pipeline


def get_row_generators(rng: Generator) -> Dict[str, Callable]:
    return {
        "age": lambda: rng.normal(54, 9),
        "trestbps": lambda: rng.normal(131, 18),
        "chol": lambda: rng.normal(246, 52),
        "thalach": lambda: rng.normal(150, 23),
        "oldpeak": lambda: rng.uniform(0, 6.2),
        "thal": lambda: rng.integers(0, 4),
        "ca": lambda: rng.integers(0, 5),
        "slope": lambda: rng.integers(0, 3),
        "exang": lambda: rng.integers(0, 2),
        "restecg": lambda: rng.integers(0, 3),
        "fbs": lambda: rng.integers(0, 2),
        "cp": lambda: rng.integers(0, 4),
        "sex": lambda: rng.integers(0, 2),
        "target": lambda: rng.integers(0, 2),
    }


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return ["thal", "ca", "slope", "exang", "restecg", "fbs", "cp", "sex"]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(scope="session")
def target_column() -> str:
    return "target"


def generate_random_row(row_generators: Dict[str, Callable]) -> Dict[str, Union[int, float]]:
    row = {}
    for key, generator in row_generators.items():
        row[key] = generator()
    return row


@pytest.fixture(scope="session")
def dataset_filename() -> str:
    return "data.csv"


@pytest.fixture(scope="session")
def dataset_size() -> int:
    return 200


@pytest.fixture(scope="session")
def dataset_file(tmp_path_factory, dataset_filename: str, dataset_size: int) -> str:
    path = tmp_path_factory.mktemp("path")
    rng = Generator(PCG64(12345))
    data = pd.DataFrame.from_records([generate_random_row(get_row_generators(rng)) for _ in range(dataset_size)])
    dataset_path = path / dataset_filename
    data.to_csv(dataset_path, index=False)
    return str(dataset_path)


@pytest.fixture(scope="session")
def test_dataset_file(tmp_path_factory, dataset_filename: str, dataset_size: int, target_column: str) -> str:
    path = tmp_path_factory.mktemp("path")
    rng = Generator(PCG64(12345))
    data = pd.DataFrame.from_records([generate_random_row(get_row_generators(rng)) for _ in range(dataset_size)])
    data.drop(columns=[target_column, ], inplace=True)
    dataset_path = path / dataset_filename
    data.to_csv(dataset_path, index=False)
    return str(dataset_path)


@pytest.fixture(scope="session")
def dataset(dataset_file: str) -> pd.DataFrame:
    return read_data(dataset_file)


@pytest.fixture(scope="session")
def test_dataset(test_dataset_file: str) -> pd.DataFrame:
    return read_data(test_dataset_file)


@pytest.fixture(scope="session")
def features(
        dataset: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
        statistics: OrderedDict[str, Callable],
        target_column: str
) -> np.ndarray:
    projection_features = 5
    polynomial_degree = 2
    n_clusters = 2
    config = get_feature_config(target_column, categorical_features, n_clusters, numerical_features, polynomial_degree,
                                projection_features, statistics)
    pipeline = build_feature_pipeline(config)
    transformed_features = pipeline.fit_transform(extract_raw_features(dataset, config))
    return transformed_features


@pytest.fixture(scope="session")
def target(dataset: pd.DataFrame, target_column: str) -> np.ndarray:
    return dataset[target_column].values


@pytest.fixture(scope="session")
def statistics() -> OrderedDict[str, Callable]:
    return OrderedDict(sum=np.sum, var=lambda x, **kwargs: np.var(x, ddof=1, **kwargs), median=np.median,
                       mean=np.mean, std=lambda x, **kwargs: np.std(x, ddof=1, **kwargs), max=np.max, min=np.min)


def get_feature_config(
        target_column: str,
        categorical_features: List[str],
        n_clusters: int,
        numerical_features: List[str],
        polynomial_degree: int,
        projection_features: int,
        statistics: OrderedDict[str, Callable]
):
    config = FeatureConfig(
        target_column=target_column,
        random_projection_features=RandomProjectionFeaturesConfig(build=True, n_features=projection_features,
                                                                  random_state=42),
        statistical_features=StatisticalFeaturesConfig(build=True, features=list(statistics)),
        k_means_features=KMeansFeaturesConfig(build=True, random_state=42, n_clusters=n_clusters),
        polynomial_features=PolynomialFeaturesConfig(build=True, degree=polynomial_degree),
        replace_zeros=True,
        raw_features=RawFeaturesConfig(
            numeric_features=numerical_features,
            categorical_features=categorical_features
        ),
    )
    return config


@pytest.fixture(scope="session")
def metrics() -> List[str]:
    return ["accuracy", "f1", "precision", "recall"]


@pytest.fixture(scope="session")
def metrics_path(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("path") / "metrics.yaml")


@pytest.fixture(scope="session")
def model_save_path(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("path") / "model.pkl")


@pytest.fixture(scope="session")
def pipeline_save_path(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("path") / "pipeline.pkl")


@pytest.fixture(scope="session")
def metadata_path(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("path") / "metadata.pkl")


@pytest.fixture(scope="session")
def train_artifacts(
        categorical_features: List[str],
        dataset_file: str,
        metrics: List[str],
        numerical_features: List[str],
        statistics: OrderedDict[str, Callable],
        model_save_path: str,
        pipeline_save_path: str,
        metrics_path: str,
        metadata_path: str,
        target_column: str
) -> Tuple[str, str, str, str]:
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
