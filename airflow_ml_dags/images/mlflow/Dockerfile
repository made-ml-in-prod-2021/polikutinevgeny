FROM python:3.7-slim-buster

COPY wait-for-it.sh wait-for-it.sh 
RUN chmod +x wait-for-it.sh

RUN pip install mlflow boto3 pymysql
