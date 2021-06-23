import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

DATA_VOLUME = f"{os.environ['DATA_MOUNT_PATH']}:/data"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(7),
        tags=["homework3"]
) as dag:
    raw_data_base = "raw/{{ ds }}/data.csv"
    target_base = "raw/{{ ds }}/target.csv"
    raw_data_path = f"/data/{raw_data_base}"
    target_path = f"/data/{target_base}"

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
    target_sensor = FileSensor(
        filepath=f"/opt/airflow/data/{target_base}",
        task_id="target_exists"
    )

    create_run = DockerOperator(
        image="airflow-create-run",
        command=[
            "--run-name", "{{ ds }}"
        ],
        network_mode="bridge",
        task_id="docker-airflow-create-run",
        do_xcom_push=True,
        environment=docker_env,
        extra_hosts=docker_hosts,
    )

    run_id = "{{ task_instance.xcom_pull(task_ids='docker-airflow-create-run', key='return_value') }}"

    processed_data_path = "/data/processed/{{ ds }}/data.csv"
    train_data_path = "/data/processed/{{ ds }}/train_data.csv"
    train_target_path = "/data/processed/{{ ds }}/train_target.csv"
    test_data_path = "/data/processed/{{ ds }}/test_data.csv"
    test_target_path = "/data/processed/{{ ds }}/test_target.csv"

    prepare_data = DockerOperator(
        image="airflow-preprocess",
        command=[
            "--input-data-path", raw_data_path,
            "--output-data-path", processed_data_path,
            "--run-id", run_id
        ],
        network_mode="bridge",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
        environment=docker_env,
        extra_hosts=docker_hosts,
    )
    split_data = DockerOperator(
        image="airflow-split-data",
        command=[
            "--data-path", processed_data_path,
            "--target-path", target_path,
            "--train-data-path", train_data_path,
            "--train-target-path", train_target_path,
            "--test-data-path", test_data_path,
            "--test-target-path", test_target_path,
            "--run-id", run_id,
        ],
        network_mode="bridge",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
        environment=docker_env,
        extra_hosts=docker_hosts,
    )
    train = DockerOperator(
        image="airflow-train",
        command=[
            "--train-data-path", train_data_path,
            "--train-target-path", train_target_path,
            "--run-id", run_id,
            "--n-estimators", "100",
            "--criterion", "gini"
        ],
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[DATA_VOLUME],
        environment=docker_env,
        extra_hosts=docker_hosts,
    )
    evaluate = DockerOperator(
        image="airflow-evaluate",
        command=[
            "--data-path", test_data_path,
            "--target-path", test_target_path,
            "--run-id", run_id,
        ],
        network_mode="bridge",
        task_id="docker-airflow-evaluate",
        volumes=[DATA_VOLUME],
        environment=docker_env,
        extra_hosts=docker_hosts,
    )

    [data_sensor, target_sensor] >> create_run >> prepare_data >> split_data >> train >> evaluate
