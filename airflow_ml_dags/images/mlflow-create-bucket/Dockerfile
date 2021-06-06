FROM python:3.6
RUN mkdir /build/
COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt

COPY create_bucket.py /create_bucket.py

ENTRYPOINT ["python", "create_bucket.py"]
