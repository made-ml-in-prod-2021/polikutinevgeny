from pathlib import Path

import click
from sklearn.datasets import load_breast_cancer


def make_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@click.command("get_data")
@click.argument("data_path")
@click.argument("target_path")
def get_data(data_path: str, target_path: str):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    make_parent_dir(data_path)
    make_parent_dir(target_path)

    X.to_csv(data_path)
    y.to_csv(target_path)


if __name__ == '__main__':
    get_data()
