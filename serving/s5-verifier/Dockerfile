# Package the verifier pipeline (model runtime + execute/repair logic + DB access) as one image.
# This is the concrete artifact of tier-2 / BYOC: you ship the container, the platform runs it on
# dedicated GPUs. Contrast with rungs 0-2a, where you had nothing to containerize.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY setup/requirements.txt /app/setup/requirements.txt
RUN pip3 install --no-cache-dir -r setup/requirements.txt fastapi uvicorn

# app code + the frozen common harness (executor lives here)
COPY common /app/common
COPY serving/s5-verifier/pipeline.py /app/pipeline.py

# Databases are mounted at runtime (never baked into the image):
#   docker run -v /data/spider:/app/data/spider -e T2S_DATA_ROOT=/app/data/spider ...
ENV MODEL_URL=http://localhost:8001/v1 \
    MODEL_NAME=dedicated \
    T2S_DATA_ROOT=/app/data/spider \
    PYTHONPATH=/app

# The model server (vLLM) runs as a sidecar on :8001; this container serves the pipeline on :8000.
EXPOSE 8000
CMD ["uvicorn", "pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
