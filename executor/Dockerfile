# executor/Dockerfile
FROM python:3.12-slim

WORKDIR /usr/src/executor

# copy runtime requirements and install
COPY requirements.txt /usr/src/executor/requirements.txt
RUN pip install --no-cache-dir -r /usr/src/executor/requirements.txt

# copy entrypoint and make executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Entrypoint: will run /job/job.py (mounted by orchestrator) by default
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# no CMD so user may override with docker run ... <cmd>
