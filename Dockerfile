FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

COPY run_surrogates.py /app/run_surrogates.py

ENTRYPOINT ["python", "/app/run_surrogates.py"]
