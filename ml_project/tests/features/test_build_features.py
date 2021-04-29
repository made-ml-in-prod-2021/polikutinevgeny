from collections import OrderedDict
from pathlib import Path
from typing import List, Callable

import numpy as np
import pandas as pd
from scipy.special import binom

from heart_disease.entities.feature_config import FeatureConfig, RandomProjectionFeaturesConfig, \
    KMeansFeaturesConfig, StatisticalFeaturesConfig, PolynomialFeaturesConfig
from heart_disease.features.build_features import \
    StatisticalFeaturesExtractor, KMeansFeaturesExtractor
from heart_disease.features.build_features import \
    build_categorical_feature_pipeline, \
    build_numerical_feature_pipeline, \
    build_feature_pipeline, \
    serialize_pipeline, \
    deserialize_pipeline, \
    extract_target
from tests.conftest import get_feature_config

EPS = 1e-6


def test_statistical_features_extractor(dataset: pd.DataFrame,
                                        statistics: OrderedDict[str, Callable]):
    extractor = StatisticalFeaturesExtractor(list(statistics))
    transformed_data = extractor.fit_transform(dataset)
    assert transformed_data.shape[0] == dataset.shape[0]
    assert transformed_data.shape[1] == len(statistics)
    for i, statistic_name in zip(range(transformed_data.shape[1]), statistics):
        assert (np.abs(transformed_data[:, i] - statistics[statistic_name](dataset.values, axis=1)) < EPS).all()


def test_kmeans_features_extractor(dataset: pd.DataFrame):
    n_clusters = 2
    extractor = KMeansFeaturesExtractor(n_clusters=n_clusters, random_state=42)
    transformed_data = extractor.fit_transform(dataset)
    assert transformed_data.shape[0] == dataset.shape[0]
    assert transformed_data.shape[1] == n_clusters
    assert np.all((transformed_data == 0) | (transformed_data == 1))


def test_categorical_feature_pipeline(
        dataset: pd.DataFrame,
        categorical_features: List[str],
):
    config = FeatureConfig()
    pipeline = build_categorical_feature_pipeline(config)
    transformed_features = pipeline.fit_transform(dataset[categorical_features])
    assert dataset.shape[0] == transformed_features.shape[0]
    n_columns = dataset[categorical_features].nunique().sum()
    assert n_columns == transformed_features.shape[1]


def test_numeric_feature_pipeline(
        dataset: pd.DataFrame,
        numerical_features: List[str],
        statistics: OrderedDict[str, Callable]
):
    projection_features = 5
    polynomial_degree = 2
    n_clusters = 2
    config = FeatureConfig(
        random_projection_features=RandomProjectionFeaturesConfig(build=True, n_features=projection_features,
                                                                  random_state=42),
        statistical_features=StatisticalFeaturesConfig(build=True, features=list(statistics)),
        k_means_features=KMeansFeaturesConfig(build=True, random_state=42, n_clusters=n_clusters),
        polynomial_features=PolynomialFeaturesConfig(build=True, degree=polynomial_degree),
        replace_zeros=True
    )
    pipeline = build_numerical_feature_pipeline(config)
    transformed_features = pipeline.fit_transform(dataset[numerical_features])
    assert transformed_features.shape[0] == dataset.shape[0]
    assert transformed_features.shape[1] == (
            projection_features +
            binom(len(numerical_features) + polynomial_degree, polynomial_degree) - 1 +
            n_clusters +
            len(statistics)
    )


def test_pipeline(
        dataset: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        statistics: OrderedDict[str, Callable],
        target_column: str
):
    projection_features = 5
    polynomial_degree = 2
    n_clusters = 2
    config = get_feature_config(target_column, categorical_features, n_clusters, numerical_features, polynomial_degree,
                                projection_features, statistics)
    pipeline = build_feature_pipeline(config)
    transformed_features = pipeline.fit_transform(dataset)
    n_cat_columns = dataset[categorical_features].nunique().sum()
    assert transformed_features.shape[0] == dataset.shape[0]
    assert transformed_features.shape[1] == (
            projection_features +
            binom(len(numerical_features) + polynomial_degree, polynomial_degree) - 1 +
            n_clusters +
            len(statistics) +
            n_cat_columns
    )


def test_extract_target(dataset, target_column: str):
    config = FeatureConfig(
        target_column=target_column
    )
    target = extract_target(dataset, config)
    assert np.all(target == dataset[target_column])


def test_serialize_pipeline(
        dataset: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        tmp_path: Path,
        statistics: OrderedDict[str, Callable],
        target_column: str
):
    filename = str(tmp_path / "pipeline.pkl")
    projection_features = 5
    polynomial_degree = 2
    n_clusters = 2
    config = get_feature_config(target_column, categorical_features, n_clusters, numerical_features, polynomial_degree,
                                projection_features, statistics)
    pipeline = build_feature_pipeline(config)
    pipeline.fit(dataset)
    serialize_pipeline(pipeline, filename)
    loaded_pipeline = deserialize_pipeline(filename)
    assert np.all(pipeline.transform(dataset) == loaded_pipeline.transform(dataset))
