from typing import List

import pandas as pd

from heart_disease.data.make_dataset import read_data, drop_duplicates, split_data, load_datasets
from heart_disease.entities.data_loading_config import SplittingConfig


def test_read_data(
        dataset_file: str,
        dataset_size: int,
        categorical_features: List[str],
        numerical_features: List[str],
        target_column: str
):
    dataset = read_data(dataset_file)
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape[0] == dataset_size
    assert set(dataset.columns) == {target_column, *categorical_features, *numerical_features}


def test_drop_duplicates(dataset: pd.DataFrame):
    dataset_without_duplicates = drop_duplicates(dataset)
    assert dataset.shape[0] >= dataset_without_duplicates.shape[0] > 0
    assert dataset[dataset.duplicated(keep=False)].shape[0] == 0


def test_split_data(dataset: pd.DataFrame):
    val_size = 0.2
    config = SplittingConfig(random_state=42, val_size=val_size)
    train, val = split_data(dataset, config)
    assert train.shape[0] == round(dataset.shape[0] * (1 - val_size))
    assert val.shape[0] == round(dataset.shape[0] * val_size)


def test_load_datasets(dataset_file: str):
    val_size = 0.2
    config = SplittingConfig(random_state=42, val_size=val_size)
    train, val = load_datasets(dataset_file, config)
    assert train.shape[0] > 0
    assert val.shape[0] > 0
