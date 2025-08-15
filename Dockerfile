# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/usr/local/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps commonly needed for ML builds and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy source
COPY . /app

# Create default data/output dirs
RUN mkdir -p /app/data/processed /app/outputs

# Default to bash; pipeline steps will override CMD with the script to run
ENTRYPOINT ["/usr/local/bin/python"]
CMD ["-m", "scripts.train_lstm_watsonx"]