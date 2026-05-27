FROM python:3.12-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY config.py data.py model.py train.py predict.py ./

ENV DATA_DIR=/data \
    ARTIFACTS_DIR=/artifacts \
    POLARS_MAX_THREADS=8

VOLUME ["/data", "/artifacts"]

CMD ["python", "train.py"]
