FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app:/app/ml \
    MODEL_WEIGHTS=/app/ml/artifacts/model.npz

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY app ./app
RUN mkdir -p /app/ml/artifacts

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
