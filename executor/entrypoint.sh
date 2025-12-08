#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="job.py"
# If the caller supplied a command, exec it
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if [ -f "${JOB_PATH}" ]; then
  # Run the job script; prints JSON to stdout as contract
  python "${JOB_PATH}"
else
  # no job file present -> return JSON error on stdout
  echo '{"error":"no job.py present"}'
  exit 2
fi
