import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "get_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(7),
        tags=["homework3"]
) as dag:
    get_data = DockerOperator(
        image="airflow-get-data",
        command="/data/raw/{{ ds }}/data.csv /data/raw/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-get-data",
        do_xcom_push=False,
        volumes=[f"{os.environ['DATA_MOUNT_PATH']}:/data"]
    )
