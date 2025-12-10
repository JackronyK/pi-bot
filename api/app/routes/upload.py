# api/app/routes/upload.py
import os
import uuid
import time
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import redis
from rq import Queue
from worker import tasks as worker_tasks

router = APIRouter()

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
redis_conn = redis.from_url(REDIS_URL, decode_responses=True)
rq_queue = Queue("default", connection=redis_conn)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/temp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _create_job_record(job_id: str, question: str = "", file_path: str = "", file_name: str = ""):
    job = {
        "job_id": job_id,
        "question": question,
        "file_path": file_path,
        "file_name": file_name,
        "status": "pending",
        "create_at": time.time(),
    }
    redis_conn.set(f"job:{job_id}", json.dumps(job))
    return job

@router.post("/upload")
async def upload_and_enqueue(file: UploadFile = File(...)):
    """ 
    Upload file and enqueue OCR job.
    Returns job_id. Processing happpens asynchronously by worker.    
    """
    # persist file to disk
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)

    try:
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # create job record
    _create_job_record(job_id, question="", file_path=path, file_name=file.filename)

    # enqueue worker.tasks.ocr_task
    # ensure worker.tasks module is importable in RQ context (PYTHONPATH)
    job = rq_queue.enqueue(worker_tasks.ocr_task, job_id, job_timeout=600, result_ttl=3600)    
    return JSONResponse({"job_id": job_id, "enqueued": True})

@router.post("/submit-file")
async def upload_and_process_sync(file: UploadFile = File(...)):
    """ 
    Uplaod file and process immediately (synchronous). useful for quick tests.
    Returns OCR result in  responce body. 
    """
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)

    try:
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # create job record
    _create_job_record(job_id, file_path=path, file_name=file.filename)

    # synchronous processing (useful for quick dev/diagnostics)
    try:
        result = worker_tasks.ocr_task(job_id)

    except Exception as e:
        return JSONResponse({"job_id": job_id, "error": str(e)}, status_code=500)
    return JSONResponse({"job_id": job_id, "result": result})