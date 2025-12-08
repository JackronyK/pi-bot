# api/app/routes/solve.py
import uuid
import json
import redis
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rq import Queue

router = APIRouter()
redis_conn = redis.from_url("redis://redis:6379", decode_responses=True)
q = Queue("default", connection=redis_conn)

class TextSolveRequest(BaseModel):
    question: str
    priority: str | None = "normal"


@router.post("/solve/text")
async def solve_text(req: TextSolveRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    
    job_id = str(uuid.uuid4())
    key = f"job:{job_id}"

    # initial job record
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "question": req.question.strip(),
        "created_at": time.time(),
        "result": None,
        "error": None,
    }

    # store  job in Redis as JSON
    redis_conn.set(key, json.dumps(job_record))

    # Enqueue worker task by import path string so the function is importable on worker side
    # the worker.tasks.process_text_job must exist
    q.enqueue("tasks.process_text_job", job_id)

    return {"job_id": job_id, "status": "queued"}

@router.get("/job/{job_id}")
async def get_job(job_id: str):
    key = f"job:{job_id}"
    raw = redis_conn.get(key)
    if raw is None:
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(raw)
