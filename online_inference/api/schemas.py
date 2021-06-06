from pathlib import Path
from typing import Union, List

from pydantic import BaseModel, BaseSettings


class HeartDiseaseModel(BaseModel):
    ids: List[int]
    features: List[List[Union[int, float]]]
    columns: List[str]


class HeartDiseaseResponseModel(BaseModel):
    id: int
    has_disease: bool


class Settings(BaseSettings):
    model_path: Path
    pipeline_path: Path
    metadata_path: Path
