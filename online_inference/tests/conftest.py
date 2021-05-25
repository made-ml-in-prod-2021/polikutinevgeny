from typing import Union, Dict, Callable, List

import pandas as pd
import pytest
from numpy.random import Generator, PCG64


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
    }


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return ["thal", "ca", "slope", "exang", "restecg", "fbs", "cp", "sex"]


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
def test_dataset(tmp_path_factory, dataset_filename: str, dataset_size: int) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("path")
    rng = Generator(PCG64(12345))
    data = pd.DataFrame.from_records([generate_random_row(get_row_generators(rng)) for _ in range(dataset_size)])
    return data
