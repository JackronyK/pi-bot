# worker/Dockerfile
# Base image: slim Python to keep it small while allowing apt installs
FROM python:3.12-slim

# -------------------------
# Metadata & working dir
# -------------------------
# working directory where worker code will be copied
WORKDIR /usr/src/worker

# make this folder importable as top-level modules
ENV PYTHONPATH=/usr/src/worker

# -------------------------
# Install system dependencies
# -------------------------
# We need:
#  - build-essential: for compiling any Python wheels if needed
#  - ca-certificates / curl / gnupg: networking & package verification
#  - poppler-utils: provides `pdftoppm` used by pdf2image for PDF -> image conversions
#  - tesseract-ocr: native Tesseract binary (math/text OCR fallback)
#  - ffmpeg: audio preprocessing (for voice notes / whisper)
#  - libraries for OpenCV / Pillow to import correctly in a headless container
#  - docker.io: docker client (so worker can call `docker` if you mount /var/run/docker.sock)
# Note: installing docker.io increases image size; remove it if you do not want the worker to call docker.
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
    libsm6 libxext6 libxrender1 \
    libjpeg-dev zlib1g-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*


# copy package and install editable, so 'import worker' works in site-packages
COPY . /usr/src/worker
RUN pip install --no-cache-dir -e /usr/src/worker

# -------------------------
# Copy and install Python requirements
# -------------------------
# Copy requirements then install in one layer for caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy worker code
# -------------------------
# Copy the worker package into the image so it is importable (PYTHONPATH set above)
COPY . .

# Start the rq worker that listens on 'default' queue and uses the Redis URL
CMD ["python", "worker_entry.py"]

