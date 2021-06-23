from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd


@click.command("predict")
@click.option("--input-path", required=True)
@click.option("--output-path", required=True)
def predict(input_path: str, output_path: str):
    data = pd.read_csv(input_path, index_col=0)

    pipeline = mlflow.sklearn.load_model(f"models:/preprocessing-pipeline/Production")
    classifier = mlflow.sklearn.load_model(f"models:/classifier-model/Production")

    data["predict"] = classifier.predict(pipeline.transform(data))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)


if __name__ == '__main__':
    predict()
