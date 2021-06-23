import click
import mlflow
import mlflow.sklearn
import pandas as pd


@click.command("evaluate")
@click.option("--data-path", required=True)
@click.option("--target-path", required=True)
@click.option("--run-id", required=True)
def evaluate(data_path: str, target_path: str, run_id: str):
    data = pd.read_csv(data_path, index_col=0)
    target = pd.read_csv(target_path, index_col=0, squeeze=True)

    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")
    classifier = mlflow.sklearn.load_model(f"runs:/{run_id}/classifier")

    prepared_data = pipeline.transform(data)

    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.eval_and_log_metrics(classifier, prepared_data, target, prefix="val_")


if __name__ == '__main__':
    evaluate()
