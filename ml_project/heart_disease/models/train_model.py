import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from heart_disease.data.make_dataset import load_datasets
from heart_disease.entities.pipeline_config import TrainingConfig
from heart_disease.features.build_features import build_feature_pipeline, extract_target, serialize_pipeline, \
    serialize_metadata, extract_raw_features
from heart_disease.models.model import train_model, evaluate_model, serialize_model, save_metrics

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainingConfig)


@hydra.main(config_path=PROJECT_ROOT / "config", config_name="train_config")
def train_pipeline(cfg: TrainingConfig):
    log.info(f"Loading data from {cfg.data_load_config.data_path}...")
    train_data, val_data = load_datasets(to_absolute_path(cfg.data_load_config.data_path),
                                         cfg.data_load_config.split_config)
    log.info("Data loaded successfully")

    log.info("Building features...")
    feature_pipeline = build_feature_pipeline(cfg.feature_config)
    raw_train_features = extract_raw_features(train_data, cfg.feature_config)
    raw_val_features = extract_raw_features(val_data, cfg.feature_config)
    feature_pipeline.fit(raw_train_features)
    train_features = feature_pipeline.transform(raw_train_features)
    val_features = feature_pipeline.transform(raw_val_features)
    train_target = extract_target(train_data, cfg.feature_config)
    val_target = extract_target(val_data, cfg.feature_config)
    log.info("Features built")

    log.info(f"Training model {cfg.model_config.model.value}...")
    model = train_model(train_features, train_target.values, cfg.model_config)
    log.info("Model trained")

    log.info("Evaluating model...")
    metrics = evaluate_model(model, val_features, val_target.values, cfg.evaluation_config.metrics)
    save_metrics(metrics, to_absolute_path(cfg.evaluation_config.metric_file_path))
    log.info("Model evaluated:")
    for metric, value in metrics.items():
        log.info(f"{metric} = {value}")

    log.info("Serializing...")
    serialize_model(model, to_absolute_path(cfg.model_save_path))
    serialize_pipeline(feature_pipeline, to_absolute_path(cfg.pipeline_save_path))
    serialize_metadata(train_data, cfg.feature_config, to_absolute_path(cfg.metadata_save_path))
    log.info("Model and pipeline serialized")


if __name__ == '__main__':
    train_pipeline()
