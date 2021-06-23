import click
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@click.command("train_model")
@click.option("--train-data-path", required=True)
@click.option("--train-target-path", required=True)
@click.option("--run-id", required=True)
@click.option("--n-estimators", required=True, type=int)
@click.option("--criterion", required=True)
def train_model(train_data_path: str,
                train_target_path: str,
                run_id: str,
                n_estimators: int,
                criterion: str):
    data = pd.read_csv(train_data_path, index_col=0)
    target = pd.read_csv(train_target_path, squeeze=True, index_col=0)

    with mlflow.start_run(run_id=run_id):
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
        clf.fit(data, target)
        mlflow.sklearn.log_model(clf, "classifier", registered_model_name="classifier-model")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)


if __name__ == '__main__':
    train_model()
