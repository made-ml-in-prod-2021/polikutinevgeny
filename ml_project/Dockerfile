FROM python:3.9.2-buster

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY heart_disease heart_disease
COPY online_inference online_inference

COPY models/model.pkl model.pkl
COPY models/pipeline.pkl pipeline.pkl
COPY models/metadata.pkl metadata.pkl

WORKDIR .

ENV model_path="/model.pkl"
ENV pipeline_path="/pipeline.pkl"
ENV metadata_path="/metadata.pkl"

CMD ["uvicorn", "online_inference.api:app", "--host", "0.0.0.0", "--port", "80"]
