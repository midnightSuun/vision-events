FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/ml \
    MODEL_WEIGHTS=/app/ml/artifacts/model.npz

WORKDIR /app

COPY detection/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ml/mlp.py ml/preprocess.py ml/infer.py /app/ml/
COPY ml/artifacts/model.npz /app/ml/artifacts/model.npz
COPY detection/main.py /app/main.py

CMD ["python", "/app/main.py"]
