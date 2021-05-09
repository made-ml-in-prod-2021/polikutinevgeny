import requests

from heart_disease.data.make_dataset import read_data

DATA_PATH = "data/heart.csv"

if __name__ == '__main__':
    data = read_data(DATA_PATH)
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "ids": list(range(data.shape[0])),
            "features": data.values.tolist(),
            "columns": data.columns.tolist()
        },
    )
    print(response.status_code)
    print(response.json())
