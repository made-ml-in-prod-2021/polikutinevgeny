from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from heart_disease.entities.splitting_config import SplittingConfig

TARGET_COLUMN = "target"


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates()


def split_data(data: pd.DataFrame, params: SplittingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data


def load_datasets(path: str, split_config: SplittingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = read_data(path)
    deduplicated_data = drop_duplicates(data)
    train_data, test_data = split_data(deduplicated_data, split_config)
    return train_data, test_data
