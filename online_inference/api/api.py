import logging
from pathlib import Path
from typing import List, Dict, Any

import cloudpickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from .schemas import HeartDiseaseModel, HeartDiseaseResponseModel, Settings

logger = logging.getLogger(__name__)

load_dotenv()
settings = Settings()
app = FastAPI(
    title="Heart disease prediction",
)


def deserialize_object(path: Path) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


@app.on_event("startup")
def load_artifacts():
    app.state.metadata = deserialize_object(settings.metadata_path)
    app.state.pipeline = deserialize_object(settings.pipeline_path)
    app.state.model = deserialize_object(settings.model_path)


def rebuild_dataframe(params: HeartDiseaseModel, metadata: Dict[str, np.dtype]) -> pd.DataFrame:
    try:
        data = pd.DataFrame(params.features, columns=params.columns)
    except ValueError:
        error_msg = "Failed to construct DataFrame from passed data"
        logger.exception(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    for key, dtype in metadata.items():
        if key not in data.columns:
            error_msg = f"Column {key} not found in data"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        if data[key].dtype != dtype:
            try:
                data[key] = data[key].astype(dtype)
            except ValueError:
                error_msg = f"Failed to cast column {key} to dtype {dtype}"
                logger.exception(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
    return data[list(metadata.keys())]


@app.post("/predict", response_model=List[HeartDiseaseResponseModel])
def predict(request: Request, params: HeartDiseaseModel):
    data = rebuild_dataframe(params, app.state.metadata)
    processed_features = request.app.state.pipeline.transform(data)
    predictions = request.app.state.model.predict(processed_features)
    return [
        HeartDiseaseResponseModel(id=id_, has_disease=pred == 1) for id_, pred in zip(params.ids, predictions)
    ]
