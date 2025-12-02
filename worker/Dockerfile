FROM python:3.12-slim

WORKDIR /usr/src/worker

# install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy worker code
COPY . .

# default command to run your worker
# (you can change this later when you implement Celery/RQ)
CMD ["python", "worker.py"]
