from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.random_projection import SparseRandomProjection

from heart_disease.entities.feature_config import FeatureConfig
from heart_disease.utils import serialize_object, deserialize_object


class StatisticalFeaturesExtractor(TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, *_, **__):
        return self

    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        return x.agg(self.features, axis=1).values


class KMeansFeaturesExtractor(TransformerMixin):
    def __init__(self, n_clusters, random_state):
        self.pipeline = make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=n_clusters, random_state=random_state)
        )
        self.ohe = OneHotEncoder(sparse=False)

    def fit(self, x, **__):
        result = np.expand_dims(self.pipeline.fit_predict(x), axis=1)
        self.ohe.fit(result)
        return self

    def transform(self, x):
        result = np.expand_dims(self.pipeline.predict(x), axis=1)
        onehot = self.ohe.transform(result)
        return onehot


def build_numerical_feature_pipeline(config: FeatureConfig) -> Pipeline:
    features = []
    if config.statistical_features.build:
        features.append(StatisticalFeaturesExtractor(config.statistical_features.features))
    features.append(PolynomialFeatures(
        degree=config.polynomial_features.degree if config.polynomial_features.build else 1,
        include_bias=False
    ))
    if config.random_projection_features.build:
        features.append(SparseRandomProjection(
            n_components=config.random_projection_features.n_features,
            random_state=config.random_projection_features.random_state
        ))
    if config.k_means_features.build:
        features.append(
            KMeansFeaturesExtractor(config.k_means_features.n_clusters, config.k_means_features.random_state))
    pipeline = []
    if config.replace_zeros:
        pipeline.append(SimpleImputer(missing_values=0, strategy="mean"))
    pipeline.append(make_union(*features))
    return make_pipeline(*pipeline, "passthrough")


def build_categorical_feature_pipeline(config: FeatureConfig) -> Pipeline:
    pipeline = [OneHotEncoder(sparse=False)]
    return make_pipeline(*pipeline, "passthrough")


def build_feature_pipeline(config: FeatureConfig) -> Pipeline:
    transformer = make_column_transformer(
        (
            build_numerical_feature_pipeline(config),
            # config.raw_features.numeric_features is not a list, but ListConfig, so scikit-learn cannot understand it
            list(config.raw_features.numeric_features)
        ),
        (
            build_categorical_feature_pipeline(config),
            list(config.raw_features.categorical_features)
        )
    )
    pipeline = [transformer]
    return make_pipeline(*pipeline, "passthrough")


def extract_target(df: pd.DataFrame, config: FeatureConfig) -> pd.Series:
    target = df[config.target_column]
    return target


def extract_raw_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    return df[config.raw_features.numeric_features + config.raw_features.categorical_features]


def serialize_pipeline(pipeline: Pipeline, path: str):
    serialize_object(pipeline, path)


def deserialize_pipeline(path: str) -> Pipeline:
    return deserialize_object(path)


def serialize_metadata(df: pd.DataFrame, config: FeatureConfig, path: str):
    all_features = config.raw_features.numeric_features + config.raw_features.categorical_features
    return serialize_object(df[all_features].dtypes.to_dict(), path)


def deserialize_metadata(path: str) -> Dict[str, np.dtype]:
    return deserialize_object(path)
