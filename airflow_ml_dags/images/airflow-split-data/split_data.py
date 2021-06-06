from pathlib import Path

import click
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE = 0.2


def make_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@click.command("split_data")
@click.option("--data-path", required=True)
@click.option("--target-path", required=True)
@click.option("--train-data-path", required=True)
@click.option("--train-target-path", required=True)
@click.option("--test-data-path", required=True)
@click.option("--test-target-path", required=True)
@click.option("--run-id", required=True)
@click.option("--test-size", type=float, default=DEFAULT_TEST_SIZE)
def split_data(
        data_path: str,
        target_path: str,
        train_data_path: str,
        train_target_path: str,
        test_data_path: str,
        test_target_path: str,
        run_id: str,
        test_size: float
):
    data = pd.read_csv(data_path, index_col=0)
    target = pd.read_csv(target_path, squeeze=True, index_col=0)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("test_size", test_size)

    for path in (train_data_path,
                 train_target_path,
                 test_data_path,
                 test_target_path):
        make_parent_dir(path)

    train_data.to_csv(train_data_path)
    test_data.to_csv(test_data_path)
    train_target.to_csv(train_target_path)
    test_target.to_csv(test_target_path)


if __name__ == '__main__':
    split_data()
