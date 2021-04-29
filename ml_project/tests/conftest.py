from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Dict, Callable

import numpy as np
import pandas as pd
import pytest
from numpy.random import Generator, PCG64

from heart_disease.data.make_dataset import read_data
from heart_disease.entities.feature_config import FeatureConfig, RandomProjectionFeaturesConfig, \
    StatisticalFeaturesConfig, KMeansFeaturesConfig, PolynomialFeaturesConfig, RawFeaturesConfig
from heart_disease.features.build_features import build_feature_pipeline


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


@pytest.fixture
def categorical_features() -> List[str]:
    return ["thal", "ca", "slope", "exang", "restecg", "fbs", "cp", "sex"]


@pytest.fixture
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture
def target_column() -> str:
    return "target"


def generate_random_row(row_generators: Dict[str, Callable]) -> Dict[str, Union[int, float]]:
    row = {}
    for key, generator in row_generators.items():
        row[key] = generator()
    return row


@pytest.fixture
def dataset_filename() -> str:
    return "data.csv"


@pytest.fixture
def dataset_size() -> int:
    return 200


@pytest.fixture
def dataset_file(tmp_path: Path, dataset_filename: str, dataset_size: int) -> str:
    rng = Generator(PCG64(12345))
    data = pd.DataFrame.from_records([generate_random_row(get_row_generators(rng)) for _ in range(dataset_size)])
    dataset_path = tmp_path / dataset_filename
    data.to_csv(dataset_path, index=False)
    return str(dataset_path)


@pytest.fixture
def dataset(dataset_file: str) -> pd.DataFrame:
    return read_data(dataset_file)


@pytest.fixture
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
    transformed_features = pipeline.fit_transform(dataset)
    return transformed_features


@pytest.fixture
def target(dataset: pd.DataFrame, target_column: str) -> np.ndarray:
    return dataset[target_column].values


@pytest.fixture
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


@pytest.fixture
def metrics() -> List[str]:
    return ["accuracy", "f1", "precision", "recall"]
