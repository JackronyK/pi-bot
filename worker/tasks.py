# worker/tasks.py
import json
import time
import redis
import os
import uuid
import logging
import tempfile
from  typing import Any, Dict, Optional

from worker.orchestrator import executor_run_docker_script
from PIL import Image, ImageOps
from pdf2image import convert_from_path
import pytesseract
from worker import utils

logger = logging.getLogger("Worker.tasks")

# Redis connection
redis_conn = redis.from_url("redis://redis:6379", decode_responses=True)


def process_text_job(job_id: str):
    """
    Worker entrypoint: process queued text job.
    - Performs static safety check (if any generated code present)
    - Supports: llm_codegen_stub -> compute actual solution via sympy but mark as llm_stub
                generic llm -> calls llm_wrapper.run_llm_stub (if present)
                docker executor -> executes code_text in sandbox (if allowed)
                fallback -> utils.local_sympy_solve_from_question
    - Writes status/result back to redis key 'job:<job_id>'
    """
    key = f"job:{job_id}"
    raw = redis_conn.get(key)
    if raw is None:
        return
    try:
        job: Dict[str, Any] = json.loads(raw)
    except Exception:
        # malformed job payload
        return

    # mark started early
    job.setdefault("started_at", time.time())
    redis_conn.set(key, json.dumps(job))

    # normalize requested solver for traceability
    requested_solver = utils.normalize_requested_solver_from_job(job)
    job.setdefault("result", {})["requested_solver"] = requested_solver
    redis_conn.set(key, json.dumps(job))

    # lazy imports & modules (fail gracefully if not available)
    try:
        import safety
    except Exception:
        safety = None
    try:
       import llm_wrapper
    except Exception:
        llm_wrapper = None
    try:
        import utils
    except Exception as e:
        # critical failure - can't solve without utils
        job["status"] = "failed"
        job["error"] = {"code": "internal_error", "message": f"Missing utils module: {e}"}
        job["finished_at"] = time.time()
        redis_conn.set(key, json.dumps(job))
        return

    # determine any generated code and its mode
    code_info = job.get("simulated_codegen") or job.get("generated_code") or {}
    code_text = code_info.get("code") if isinstance(code_info, dict) else None
    mode = code_info.get("mode", "") if isinstance(code_info, dict) else ""

    # Safety check first (if there's code)
    if code_text:
        if safety:
            try:
                safe, issues = safety.is_code_safe(code_text)
            except Exception as e:
                safe = False
                issues = [f"safety_scanner_exception: {e}"]
            if not safe:
                job["status"] = "failed"
                job["error"] = {
                    "code": "safety_reject",
                    "message": "Generated code rejected by static safety checks",
                    "issues": issues,
                }
                # annotate metadata
                job.setdefault("result", {})["requested_solver"] = requested_solver
                job.setdefault("result", {})["solved_by"] = None
                job.setdefault("result", {})["execution_path"] = "safety_reject"
                job.setdefault("result", {}).setdefault("metadata", {})["generated_code"] = code_text
                job["finished_at"] = time.time()
                redis_conn.set(key, json.dumps(job))
                return
        else:
            # Safety module not available - be conservative and reject (or change if dev)
            job["status"] = "failed"
            job["error"] = {"code": "internal_error", "message": "safety module not available"}
            job.setdefault("result", {})["requested_solver"] = requested_solver
            job.setdefault("result", {})["solved_by"] = None
            job.setdefault("result", {})["execution_path"] = "safety_reject_no_scanner"
            job["finished_at"] = time.time()
            redis_conn.set(key, json.dumps(job))
            return

    # HANDLE stub LLM mode: produce safe stub code (or use provided code) but compute real answer via SymPy
    if mode == "llm_codegen_stub":
        # if no code_text, ask generator for safe stub (optional)
        if not code_text and llm_wrapper and hasattr(llm_wrapper, "generate_stub_code"):
            try:
                gen = llm_wrapper.generate_stub_code(job.get("question", ""))
                code_text = gen.get("code")
                code_info["code"] = code_text
                job["simulated_codegen"] = code_info
            except Exception:
                code_text = None

        # compute real solution with SymPy for deterministic correctness
        try:
            result = utils.local_sympy_solve_from_question(job.get("question", ""))
        except Exception as e:
            job["status"] = "failed"
            job["error"] = {"code": "solver_error", "message": str(e)}
            job["finished_at"] = time.time()
            redis_conn.set(key, json.dumps(job))
            return

        # annotate metadata: requested LLM but solved by SymPy stub (this is expected in stub flow)
        result["solved_by"] = "sympy_stub"
        result["execution_path"] = "sympy_fallback_from_llm_stub"
        result["requested_solver"] = requested_solver
        if code_text:
            result.setdefault("metadata", {})["generated_code"] = code_text

        job["result"] = result
        job["status"] = "done"
        job["finished_at"] = time.time()
        redis_conn.set(key, json.dumps(job))
        return

    # Generic LLM path - attempt to run llm_wrapper.run_llm_stub (safe stub runner) if available.
    if mode.startswith("llm") and llm_wrapper and hasattr(llm_wrapper, "run_llm_stub"):
        try:
            # run_llm_stub SHOULD return a dict shaped like utils.local_sympy_solve_from_question
            llm_result = llm_wrapper.run_llm_stub(code_text, job.get("question", ""))
            # normalize result metadata
            llm_result.setdefault("solved_by", "llm_wrapper")
            llm_result.setdefault("execution_path", "llm_wrapper_run")
            llm_result.setdefault("requested_solver", requested_solver)
            if code_text:
                llm_result.setdefault("metadata", {})["generated_code"] = code_text
            job["result"] = llm_result
            job["status"] = "done"
            job["finished_at"] = time.time()
            redis_conn.set(key, json.dumps(job))
            return
        except Exception as e:
            job["status"] = "failed"
            job["error"] = {"code": "llm_error", "message": str(e)}
            job["finished_at"] = time.time()
            redis_conn.set(key, json.dumps(job))
            return

    # If codegen produced code that we might execute (llm_codegen) and user wants to run in docker executor:
    if mode == "llm_codegen" and code_text:
        # safety already passed above; try docker exec if available/allowed
        if allow_docker_executor := globals().get("DOCKER_RUN_ENABLED", False):
            try:
                exec_out = executor_run_docker_script(code_text, job_id)
                try:
                    parsed = json.loads(exec_out.get("stdout", "") or "{}")
                except Exception:
                    parsed = {"stdout": exec_out.get("stdout", ""), "stderr": exec_out.get("stderr", ""), "exit_code": exec_out.get("exit_code")}
                parsed.setdefault("solved_by", "docker_executor")
                parsed.setdefault("execution_path", "executed_in_docker")
                parsed.setdefault("requested_solver", requested_solver)
                if code_text:
                    parsed.setdefault("metadata", {})["generated_code"] = code_text
                job["result"] = parsed
                job["status"] = "done"
                job["finished_at"] = time.time()
                redis_conn.set(key, json.dumps(job))
                return
            except Exception as e:
                # If docker execution fails, mark as failed or fallback depending on policy
                job["status"] = "failed"
                job["error"] = {"code": "executor_error", "message": str(e)}
                job["finished_at"] = time.time()
                redis_conn.set(key, json.dumps(job))
                return
        else:
            # Docker not enabled: fall back to SymPy (safe)
            try:
                result = utils.local_sympy_solve_from_question(job.get("question", ""))
            except Exception as e:
                job["status"] = "failed"
                job["error"] = {"code": "solver_error", "message": str(e)}
                job["finished_at"] = time.time()
                redis_conn.set(key, json.dumps(job))
                return

            result["solved_by"] = "sympy_stub"
            result["execution_path"] = "sympy_fallback_from_llm_codegen"
            result["requested_solver"] = requested_solver
            if code_text:
                result.setdefault("metadata", {})["generated_code"] = code_text

            job["result"] = result
            job["status"] = "done"
            job["finished_at"] = time.time()
            redis_conn.set(key, json.dumps(job))
            return

    # Final fallback path: no LLM request or none matched -> run SymPy solver
    try:
        result = utils.local_sympy_solve_from_question(job.get("question", ""))
    except Exception as e:
        job["status"] = "failed"
        job["error"] = {"code": "solver_error", "message": str(e)}
        job["finished_at"] = time.time()
        redis_conn.set(key, json.dumps(job))
        return

    result.setdefault("solved_by", "sympy_stub")
    result.setdefault("execution_path", "sympy")
    result.setdefault("requested_solver", requested_solver)
    if code_text:
        result.setdefault("metadata", {})["generated_code"] = code_text

    job["result"] = result
    job["status"] = "done"
    job["finished_at"] = time.time()
    redis_conn.set(key, json.dumps(job))
    return


def _save_temp_uploaded_file(upload_bytes: bytes, filename_hint: str = "upload") -> str:
    """  
    Save uploaded bytes to a temporary file and return the path.
    """
    fd, temp_path = tempfile.mkstemp(prefix=f"{filename_hint}_", suffix="")
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(upload_bytes)
    return temp_path

def _image_preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """ 
       Minimal preprocessing pipeline:
      - convert to grayscale
      - resize if very large (helps tesseract)
      - apply simple auto-contrast / binarization
    Improve this pipeline as needed (deskew, denoise, morphological ops).
    """
    # convert to  grayscale
    img = img.convert("L")
    # auto-contrast
    img = ImageOps.autocontrast(img)
    # Optionally resize down/up to target width while maintaining aspect
    maxw = 2000
    if img.width > maxw:
        ration = maxw / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    return img

def extract_text_from_image(image: Image.Image) -> str:
    """Run pytesseract on a OIL.Image after preprocessing"""
    proc = _image_preprocess_for_ocr(image)
    # Configure tesseract for plain text output;
    try:
        text = pytesseract.image_to_string(proc)
    except Exception as e:
        logger.exception("pytessetact failed: %s", e)
        text = ""
    return text

def extract_text_from_pdf(pdf_path: str, dpi: int = 200, first_n_pages: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert PDF -> images, run OCR per page and collect results.
    Returns dict:
      { "pages": [ { "page_no": 1, "text": "..." }, ... ], "raw": <concatenated text> }
    """
    pages_text = []
    try:
        # convert_from_path uses poppler pdftoppm installed on host/container
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        logger.exception("pdf2image.convert_from_path failed: %s", e)
        return {"pages": [], "raw": ""}
    
    if first_n_pages:
        pil_pages = pil_pages[:first_n_pages]
    all_text = []
    for idx, pil in enumerate(pil_pages, start=1):
        text = extract_text_from_image(pil)
        pages_text.append({"page_no": idx, "text": text})
        all_text.append(text)

    return {"pages": pages_text, "raw": "\n\n".join(all_text)}

def run_math_ocr_placeholder(image_or_text: Any) -> Dict[str, Any]:
    """
    Placeholder for math OCR (pix2tex / LaTeX-OCR).
    For now returns empty latex list. Later replace with actual model call.
    """
    # TODO: integrate pix2tex / LaTex-OCR here
    return {"latex": [], "confidence": 0.0}

def ocr_task(job_id: str):
    """
    RQ worker entry for OCR jobs.
    Expected job record (in Redis) to contain:
      - job_id
      - file_path OR 'upload_bytes' (base64 not used here; API saves file to path and records file_path)
      - file_name (optional)
    The task will:
      1. Update job.status -> running
      2. Read file (pdf or image) and extract text
      3. Optionally run math OCR placeholder
      4. Save result into job.result and set status done
    """
    key = f"job:{job_id}"
    raw = redis_conn.get(key) if redis_conn else None
    if raw is None:
        logger.warning("ocr_task: job %s not found in redis", job_id)
        return 
    
    job = json.loads(raw)
    utils.update_job_record(job_id, {"status": "running", "started_at": time.time()})

    try:
        file_path = job.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"file_path missing or does not exist: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        result = {}

        if ext in (".pdf", ):
            # PDF path -> convert pages -> OCR
            ocr_output = extract_text_from_pdf(file_path, dpi=200, first_n_pages=None)
            result["pages"] = ocr_output.get("pages", [])
            result["raw_text"] = ocr_output.get("raw", "")
        else:
            # single image file
            try:
                img = Image.open(file_path)
                t = extract_text_from_image(img)
                result["pages"] = [{"page_no": 1, "text": t}]
                result["raw_text"] = t
            except Exception as e:
                logger.exception("Failed to open/process image %s: %s", file_path, e)
                result["pages"] = []
                result["raw_text"] = ""
        # run math OCR placeholder
        math_out = run_math_ocr_placeholder(result["raw_text"] or result['pages'])
        result["math"] = math_out

        # Pack solver-ready payload if desired (e.g., put extracted_text in 'question' for orchestration)
        # For now store OCR-only result
        ocr_result = {
            "job_id": job_id,
            "extracted_text": result['raw_text'],
            "pages": result['pages'],
            "math": result["math"],
            "finished_at": time.time(),
        }

        # update job record and mark done
        utils.update_job_record(job_id, {"status": "done", "finished_at": time.time(), "result": ocr_result})
        return ocr_result
    
    except Exception as e:
        logger.exception("ocr_task failed for job %s: %s", job_id, e)
        utils.update_job_record(job_id, {"status": "failed", "finished_at": time.time(), "error": {'message': str(e)}})
        return {"error": str(e)}


