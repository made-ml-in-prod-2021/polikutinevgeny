from typing import Tuple, List

import pandas as pd
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pydantic import parse_obj_as

from online_inference.api.schemas import HeartDiseaseModel, HeartDiseaseResponseModel


@pytest.fixture(scope="session")
def test_client(tmp_path_factory, train_artifacts: Tuple[str, str, str, str]):
    # Alas, FastAPI currently cannot use DI in events.
    # Setting up .env config manually
    # https://github.com/tiangolo/fastapi/issues/2057

    path = tmp_path_factory.mktemp("path")

    _, model_save_path, pipeline_save_path, metadata_path = train_artifacts
    new_env = path / ".env"
    with open(new_env, "w") as f:
        print(f"model_path={model_save_path}", file=f)
        print(f"pipeline_path={pipeline_save_path}", file=f)
        print(f"metadata_path={metadata_path}", file=f)
    load_dotenv(dotenv_path=new_env)

    from online_inference.api.api import app
    client = TestClient(app)
    return client


def test_predict(test_dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(test_dataset.shape[0]))
    request = HeartDiseaseModel(
        ids=ids,
        features=test_dataset.values.tolist(),
        columns=test_dataset.columns.tolist()
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 200
    preds = parse_obj_as(List[HeartDiseaseResponseModel], response.json())
    assert len(preds) == test_dataset.shape[0]
    assert set([i.id for i in preds]) == set(ids)


def test_predict_wrong_shape(test_dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(test_dataset.shape[0]))
    request = HeartDiseaseModel(
        ids=ids,
        features=test_dataset.values.tolist(),
        columns=test_dataset.columns.tolist()[:-1]
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400


def test_predict_wrong_column(test_dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(test_dataset.shape[0]))
    columns = test_dataset.columns.tolist()[:-1] + ["obviously_extra_column"]
    request = HeartDiseaseModel(
        ids=ids,
        features=test_dataset.values.tolist(),
        columns=columns
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400


def test_predict_wrong_dtype(test_dataset: pd.DataFrame, test_client: TestClient, categorical_features: List[str]):
    dataset_copy = test_dataset.copy(deep=True)
    ids = list(range(dataset_copy.shape[0]))
    dataset_copy[categorical_features[0]] = float('nan')
    request = HeartDiseaseModel(
        ids=ids,
        features=dataset_copy.values.tolist(),
        columns=dataset_copy.columns.tolist()
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400
