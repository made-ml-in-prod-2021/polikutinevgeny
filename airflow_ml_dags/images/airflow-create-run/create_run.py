import click
import mlflow


@click.command("create_run")
@click.option("--run-name")
def create_run(run_name: str):
    with mlflow.start_run(run_name=run_name):
        current_run_id = mlflow.active_run().info.run_id
    print("\n", current_run_id)  # output to XCom


if __name__ == '__main__':
    create_run()
