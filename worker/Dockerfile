# worker/Dockerfile
FROM python:3.12-slim

# working directory where worker code will be copied
WORKDIR /usr/src/worker

# make this folder importable as top-level modules
ENV PYTHONPATH=/usr/src/worker

# install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    docker.io \
    && rm -rf /var/lib/apt/lists/*


# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy worker package code (so worker.tasks is importable)
COPY . .

# Start the rq worker that listens on 'default' queue and uses the Redis URL
CMD ["python", "worker_entry.py"]

