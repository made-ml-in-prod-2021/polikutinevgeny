FROM python:3.9.2-buster AS build

COPY ml_project ml_project
WORKDIR /ml_project
RUN pip install --upgrade build
RUN python -m build

FROM python:3.9.2-buster

COPY online_inference/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY --from=build /ml_project/dist dist
RUN pip install dist/*.whl && rm -rf dist

COPY online_inference/api api

COPY ml_project/models/model.pkl model.pkl
COPY ml_project/models/pipeline.pkl pipeline.pkl
COPY ml_project/models/metadata.pkl metadata.pkl

WORKDIR .

ENV model_path="/model.pkl"
ENV pipeline_path="/pipeline.pkl"
ENV metadata_path="/metadata.pkl"

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "80"]
