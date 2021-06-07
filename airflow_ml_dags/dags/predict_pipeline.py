import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict_pipeline",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(7),
        tags=["homework3"]
) as dag:
    raw_data_base = "raw/{{ ds }}/data.csv"
    raw_data_path = f"/data/{raw_data_base}"

    docker_env = {
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "MLFLOW_S3_ENDPOINT_URL": os.environ["MLFLOW_S3_ENDPOINT_URL"],
        "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
    }
    docker_hosts = {
        "host.docker.internal": "host-gateway"
    }

    data_sensor = FileSensor(
        filepath=f"/opt/airflow/data/{raw_data_base}",
        task_id="data_exists"
    )

    evaluate = DockerOperator(
        image="airflow-predict",
        command=[
            "--input-path", raw_data_path,
            "--output-path", "/data/predictions/{{ ds }}/predictions.csv",
        ],
        network_mode="bridge",
        task_id="docker-airflow-evaluate",
        volumes=[f"{os.environ['DATA_MOUNT_PATH']}:/data"],
        environment=docker_env,
        extra_hosts=docker_hosts,
    )

    data_sensor >> evaluate
