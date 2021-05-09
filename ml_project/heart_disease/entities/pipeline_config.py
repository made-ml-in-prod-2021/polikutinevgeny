from dataclasses import dataclass, field

import omegaconf

from heart_disease.entities.data_loading_config import DataLoadingConfig
from heart_disease.entities.feature_config import FeatureConfig
from heart_disease.entities.model_config import TrainModelConfig, EvaluateModelConfig


@dataclass
class TrainingConfig:
    data_load_config: DataLoadingConfig = field(default_factory=lambda: DataLoadingConfig)
    feature_config: FeatureConfig = field(default_factory=lambda: FeatureConfig)
    model_config: TrainModelConfig = field(default_factory=lambda: TrainModelConfig)
    evaluation_config: EvaluateModelConfig = field(default_factory=lambda: EvaluateModelConfig)
    model_save_path: str = omegaconf.MISSING
    pipeline_save_path: str = omegaconf.MISSING
    metadata_save_path: str = omegaconf.MISSING


@dataclass
class PredictConfig:
    model_load_path: str = omegaconf.MISSING
    pipeline_load_path: str = omegaconf.MISSING
    data_path: str = omegaconf.MISSING
    output_path: str = omegaconf.MISSING
