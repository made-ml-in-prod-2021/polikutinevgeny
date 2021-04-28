import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.random_projection import SparseRandomProjection

from heart_disease.entities.feature_config import FeatureConfig


class StatisticalFeaturesExtractor(TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self, *_, **__):
        return self

    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        return x.agg(self.features, axis=1)


class KMeansFeaturesExtractor(TransformerMixin):
    def __init__(self, random_state):
        self.pipeline = make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=2, random_state=random_state)
        )

    def fit(self, x, **__):
        self.pipeline.fit(x)
        return self

    def transform(self, x):
        result = np.expand_dims(self.pipeline.predict(x), axis=1)
        return result


class PassthroughTransformer(TransformerMixin):
    def fit(self, *_, **__):
        return self

    def transform(self, x):
        return x


def build_numerical_feature_pipeline(config: FeatureConfig) -> Pipeline:
    features = [PassthroughTransformer()]
    if config.statistical_features.build:
        features.append(StatisticalFeaturesExtractor(config.statistical_features.features))
    if config.polynomial_features.build:
        features.append(PolynomialFeatures(config.polynomial_features.degree))
    if config.random_projection_features.build:
        features.append(SparseRandomProjection(
            n_components=config.random_projection_features.n_features,
            random_state=config.random_projection_features.random_state
        ))
    if config.k_means_features.build:
        features.append(KMeansFeaturesExtractor(config.k_means_features.random_state))
    pipeline = []
    if config.replace_zeros:
        pipeline.append(SimpleImputer(missing_values=0, strategy="mean"))
    pipeline.append(make_union(*features))
    return make_pipeline(*pipeline, "passthrough")


def build_categorical_feature_pipeline(config: FeatureConfig) -> Pipeline:
    pipeline = [OneHotEncoder()]
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
    if config.n_features_to_select > 0:
        pipeline.append(SelectKBest(score_func=mutual_info_classif, k=config.n_features_to_select))
    return make_pipeline(*pipeline, "passthrough")


def extract_target(df: pd.DataFrame, config: FeatureConfig):
    target = df[config.target_column]
    return target


def serialize_pipeline(pipeline: Pipeline, path: str):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def deserialize_pipeline(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
