# Stage 1: Build
FROM python:3.12.13-slim-bookworm AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Finish
FROM python:3.12.13-slim-bookworm

RUN groupadd -r appuser && useradd -r -g appuser -M -s /sbin/nologin appuser 

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER appuser

WORKDIR /app

COPY mil-yolov26n.onnx mil-yolov26n.onnx
COPY --chown=appuser:appuser helpers.py helpers.py
COPY --chown=appuser:appuser vision.py vision.py
COPY --chown=appuser:appuser main.py main.py

CMD ["python", "main.py"]
