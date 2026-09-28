FROM python:3.11-slim

WORKDIR /app

# Keep native CPU libraries from creating an unbounded worker pool when the
# analyzer invokes the local model from its ingestion worker.  The value can be
# overridden at deployment time after the instance has been sized accordingly.
ENV TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# The PyPI Linux wheel pulls CUDA libraries even though this deployment runs on
# CPU. Install the official CPU wheel first; the requirements constraint then
# reuses it instead of adding several gigabytes of unused NVIDIA packages.
RUN pip install --no-cache-dir torch==2.14.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN mkdir -p /app/data/datasets /app/data/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["ddtrace-run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
