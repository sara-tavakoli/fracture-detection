# Multi-stage build for the inference service.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS builder
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install \
        -r requirements.txt \
        "fastapi>=0.111" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" "pydicom>=2.4"
COPY src ./src
RUN pip install --prefix=/install --no-deps .

FROM base AS runtime
COPY --from=builder /install /usr/local
COPY src ./src
COPY configs ./configs
ENV FRACTURE_CKPT=/app/artifacts/best.ckpt \
    FRACTURE_DEVICE=cpu
EXPOSE 8000
# artifacts/ (the trained checkpoint) is mounted at run time:
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts fracture
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "fracture.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
