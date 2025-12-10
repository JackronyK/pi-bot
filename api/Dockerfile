# api/Dockerfile
FROM python:3.12-slim


# keep image small and deterministic
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set working directory for the api app
WORKDIR /usr/src/app

#  system deps that might be needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# copy api requirements and install them
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the worker package into the image and install editable
# (this allows imports like `import worker` to resolve from site-packages)
COPY ../worker /usr/src/worker
RUN pip install --no-cache-dir -e /usr/src/worker

# copy app source
COPY . .

# default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]