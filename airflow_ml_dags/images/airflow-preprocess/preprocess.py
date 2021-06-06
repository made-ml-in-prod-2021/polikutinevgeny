from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


def get_pipeline() -> Pipeline:
    return make_pipeline(
        SimpleImputer(),
        StandardScaler()
    )


@click.command("preprocess")
@click.option("--input-data-path", required=True)
@click.option("--output-data-path", required=True)
@click.option("--run-id", required=True)
def preprocess(input_data_path: str, output_data_path: str, run_id: str):
    data = pd.read_csv(input_data_path, index_col=0)

    with mlflow.start_run(run_id=run_id):
        pipeline = get_pipeline()
        prepared_data = pipeline.fit_transform(data)
        mlflow.sklearn.log_model(pipeline, "pipeline", registered_model_name="preprocessing-pipeline")

    Path(output_data_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prepared_data, columns=data.columns, index=data.index).to_csv(output_data_path)


if __name__ == '__main__':
    preprocess()
