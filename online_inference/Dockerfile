FROM python:3.9.2-buster

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api api

COPY model.pkl model.pkl
COPY pipeline.pkl pipeline.pkl
COPY metadata.pkl metadata.pkl

WORKDIR .

ENV model_path="/model.pkl"
ENV pipeline_path="/pipeline.pkl"
ENV metadata_path="/metadata.pkl"

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "80"]
