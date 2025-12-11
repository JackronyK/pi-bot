# worker/Dockerfile
# Base image: slim Python to keep it small while allowing apt installs
FROM python:3.12-slim

# -------------------------
# Metadata & working dir
# -------------------------
# Keep code in /usr/src so package path resolution works:
WORKDIR /usr/src

# -------------------------
# Install system dependencies
# -------------------------
# We install poppler-utils (pdfinfo/pdftoppm), tesseract, ffmpeg and minimal libs
# so OCR and PDF->image conversion run inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    zlib1g-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*


# -------------------------
# Copy code & install package
# -------------------------
# Copy worker sources into /usr/src/worker (project root) and install editable.
# Installing editable helps local dev, but we keep PYTHONPATH correct so imports work.
COPY . /usr/src/worker

# Install the worker package in editable mode so "pip show ..." works and
# packaging metadata is created. This is helpful for dev/test; remove if prefer no editable install.
RUN pip install --no-cache-dir -e /usr/src/worker


# Install Python requirements (if you prefer requirements first for layer caching,
# move COPY requirements & pip install lines before COPY .)
COPY requirements.txt /tmp/requirements-worker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-worker.txt && rm /tmp/requirements-worker.txt

# -------------------------
# Runtime environment
# -------------------------
# Make the parent directory /usr/src on PYTHONPATH so `import worker` finds /usr/src/worker
ENV PYTHONPATH=/usr/src:${PYTHONPATH:-}


# Default command: start worker entrypoint (RQ worker)
WORKDIR /usr/src/worker
CMD ["python", "worker_entry.py"]

