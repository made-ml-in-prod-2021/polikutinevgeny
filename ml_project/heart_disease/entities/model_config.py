from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

import omegaconf


class ModelType(Enum):
    random_forest = "RandomForestClassifier"
    extra_trees = "ExtraTreesClassifier"


@dataclass
class TrainModelConfig:
    model: ModelType = field(default=ModelType.random_forest)
    random_state: int = field(default=42)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluateModelConfig:
    metric_file_path: str = omegaconf.MISSING
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
