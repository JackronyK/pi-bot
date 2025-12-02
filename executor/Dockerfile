FROM python:3.12-slim

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    build-essential \
    libgl1 \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/executor
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy executor runner script
COPY run_executor.py .

#Default command
CMD [ "python", "run_executor.py" ]

