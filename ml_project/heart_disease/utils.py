import cloudpickle
from typing import Any

cloudpickle.register_deep_serialization("heart_disease")


def serialize_object(obj: Any, path: str):
    with open(path, "wb") as f:
        cloudpickle.dump(obj, f)


def deserialize_object(path: str) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)
