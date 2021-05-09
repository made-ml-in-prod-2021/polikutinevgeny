import pickle
from typing import Any


def serialize_object(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def deserialize_object(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
