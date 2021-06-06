from pathlib import Path

from heart_disease.utils import serialize_object, deserialize_object


def test_serialize_object(tmp_path: Path):
    path = str(tmp_path / "path.pkl")
    objects = [1, 2.0, "test", {1, 2, "test2"}, {"hello": "there", "general": 42}, ["one", 2, 3.0]]
    for obj in objects:
        serialize_object(obj, path)
        assert deserialize_object(path) == obj
