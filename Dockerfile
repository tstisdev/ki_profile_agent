FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.cache/huggingface && chmod -R 755 /root/.cache/huggingface

COPY . .

RUN mkdir -p /app/data /app/storage

ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

## Remove eager model download to speed up builds; cache via volume at runtime

CMD ["python","src/main.py"]