from dataclasses import dataclass, field
from typing import List


@dataclass
class StatisticalFeaturesConfig:
    build: bool = field(default=False)
    features: List[str] = field(default_factory=lambda: ["sum", "var", "median", "mean", "std", "max", "min"])


@dataclass
class RandomProjectionFeaturesConfig:
    build: bool = field(default=False)
    n_features: int = field(default=5)
    random_state: int = field(default=42)


@dataclass
class PolynomialFeaturesConfig:
    build: bool = field(default=False)
    degree: int = field(default=2)


@dataclass
class KMeansFeaturesConfig:
    build: bool = field(default=False)
    random_state: int = field(default=42)


@dataclass
class RawFeaturesConfig:
    numeric_features: List[str] = field(default_factory=lambda: ["age", "trestbps", "chol", "thalach", "oldpeak"])
    categorical_features: List[str] = field(
        default_factory=lambda: ["thal", "ca", "slope", "exang", "restecg", "fbs", "cp", "sex"])


@dataclass
class FeatureConfig:
    target_column: str = field(default="target")

    raw_features: RawFeaturesConfig = field(default_factory=RawFeaturesConfig)
    statistical_features: StatisticalFeaturesConfig = field(default_factory=StatisticalFeaturesConfig)
    random_projection_features: RandomProjectionFeaturesConfig = field(default_factory=RandomProjectionFeaturesConfig)
    polynomial_features: PolynomialFeaturesConfig = field(default_factory=PolynomialFeaturesConfig)
    k_means_features: KMeansFeaturesConfig = field(default_factory=KMeansFeaturesConfig)

    replace_zeros: bool = field(default=True)

    n_features_to_select: int = field(default=-1)
