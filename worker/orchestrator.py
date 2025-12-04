# worker/orchestrator.py

"""
Job Orchestrator with optional LLM integration.

Behavior:
- Uses LLM wrapper if PIBOT_USE_LLM=true; else uses local SymPy solver.
- Supports optional docker-run execution for generated code.
- Persists job status/results in Redis.
"""

import json
import os
import time
import tempfile
import shutil
import subprocess
import logging
from typing import Any, Dict

import redis
import utils, safety

logger = logging.getLogger("orchestrator")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
redis_conn = redis.from_url(REDIS_URL, decode_responses=True)

# Docker executor configuration
EXECUTOR_IMAGE = os.environ.get("PIBOT_EXECUTOR_IMAGE", "pibot-executor:latest")
DOCKER_RUN_ENABLED = os.environ.get("PIBOT_DOCKER_RUN", "false").lower() in ("1", "true", "yes")
DOCKER_TIMEOUT_SECONDS = int(os.environ.get("PIBOT_DOCKER_TIMEOUT", "8"))
DOCKER_MEMORY = os.environ.get("PIBOT_DOCKER_MEMORY", "512m")
DOCKER_CPUS = os.environ.get("PIBOT_DOCKER_CPUS", "0.5")

# LLM wrapper optional import
PIBOT_USE_LLM = os.environ.get("PIBOT_USE_LLM", "false").lower() in ("1", "true", "yes")
if PIBOT_USE_LLM:
    try:
        import worker.llm_wrapper as llm_wrapper
    except Exception:
        logger.exception("Failed to import llm_wrapper")
        llm_wrapper = None
else:
    llm_wrapper = None


# ---------------------------
# Docker executor
# ---------------------------
def executor_run_docker_script(script_text: str, job_id: str, timeout: int = DOCKER_TIMEOUT_SECONDS) -> Dict[str, Any]:
    if not DOCKER_RUN_ENABLED:
        raise RuntimeError("Docker-run executor disabled.")

    workdir = tempfile.mkdtemp(prefix=f"pibot_job_{job_id}_")
    script_path = os.path.join(workdir, "job.py")
    with open(script_path, "w", encoding="utf8") as f:
        f.write(script_text)

    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", DOCKER_MEMORY,
        "--cpus", DOCKER_CPUS,
        "-v", f"{workdir}:/job:ro",
        EXECUTOR_IMAGE,
        "bash", "-lc",
        f"timeout {timeout}s python /job/job.py"
    ]

    try:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout + 2)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc = "", f"timeout expired: {e}", 124
    except Exception as e:
        stdout, stderr, rc = "", f"executor error: {e}", 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return {"stdout": stdout, "stderr": stderr, "exit_code": rc}


# ---------------------------
# Orchestration
# ---------------------------
def orchestrate_job(job_id: str, allow_docker_executor: bool = False) -> Dict[str, Any]:
    job = utils.get_job_record(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    question = job.get("question", "")
    utils.update_job_record(job_id, {"status": "running", "started_at": time.time()})

    requested_solver = utils._normalize_requested_solver_from_job(job)
    utils.update_job_record(job_id, {"result": {"requested_solver": requested_solver}})

    # ---------------------------
    # Step 1: Parse question
    # ---------------------------
    structured = None
    if PIBOT_USE_LLM and llm_wrapper:
        try:
            structured = llm_wrapper.parse_text_to_json(question)
        except Exception:
            structured = None

    if structured is None:
        eqs = utils.extract_equations(question)
        structured = {
            "raw_question": question,
            "equations": eqs,
            "unknowns": utils.detect_unknowns(eqs),
            "goal": "solve",
            "notes": "heuristic-fallback",
        }

    # ---------------------------
    # Step 2: Codegen (LLM or sympy stub)
    # ---------------------------
    codegen_out = {"mode": "sympy_stub", "code": None, "metadata": {}}
    if PIBOT_USE_LLM and llm_wrapper:
        try:
            codegen_out = llm_wrapper.generate_python_from_json(structured)
        except Exception as e:
            codegen_out = {"mode": "sympy_stub", "code": None, "metadata": {"error": str(e)}}

    result = None
    if codegen_out.get("mode") == "llm_codegen" and codegen_out.get("code"):
        code_text = codegen_out["code"]

        # Safety check
        safe, issues = False, []
        try:
            safe, issues = safety.is_code_safe(code_text)
        except Exception as e:
            issues = [f"safety_scanner_error: {e}"]

        if not safe:
            utils.update_job_record(job_id, {
                "status": "failed",
                "finished_at": time.time(),
                "error": {"code": "safety_reject", "message": "Code rejected", "issues": issues},
                "result": {"requested_solver": requested_solver, "solved_by": None, "execution_path": "safety_reject"}
            })
            return {"error": "safety_reject", "issues": issues}

        if allow_docker_executor and DOCKER_RUN_ENABLED:
            exec_out = executor_run_docker_script(code_text, job_id)
            try:
                parsed = json.loads(exec_out.get("stdout") or "{}")
            except Exception:
                parsed = exec_out
            parsed.update({"solved_by": "docker_executor", "execution_path": "executed_in_docker", "requested_solver": requested_solver})
            utils.update_job_record(job_id, {"status": "done", "finished_at": time.time(), "result": parsed})
            return parsed
        else:
            result = utils.local_sympy_solve_from_question(question)
            result.update({"solved_by": "sympy_stub", "execution_path": "sympy_fallback_from_llm_codegen", "requested_solver": requested_solver, "metadata": {"generated_code": code_text}})
    else:
        # SymPy fallback path
        result = utils.local_sympy_solve_from_question(question)
        result.update({"solved_by": "sympy_stub", "execution_path": "sympy", "requested_solver": requested_solver})

    # ---------------------------
    # Step 3: Optional LLM explanation
    # ---------------------------
    if PIBOT_USE_LLM and llm_wrapper:
        try:
            expl = llm_wrapper.explain_result(question, result)
            if expl and expl.get("steps_html"):
                result["steps_html"] = expl["steps_html"]
                result.setdefault("explain_meta", {})["from_llm"] = True
        except Exception:
            pass

    # ---------------------------
    # Step 4: Validation
    # ---------------------------
    try:
        eqs = structured.get("equations", [])
        if eqs and result:
            eq_objs = utils.build_sympy_eqs(eqs)
            unk = structured.get("unknowns", utils.detect_unknowns(eqs))
            from sympy import symbols
            symb = symbols(" ".join(unk) if unk else "x")
            valid = utils.validate_solution(eq_objs, result.get("structured", {}).get("solutions", result.get("answer")), symb if isinstance(symb, (list, tuple)) else [symb])
            result.setdefault("validation", {})["ok"] = bool(valid)
    except Exception:
        pass  # best-effort validation

    utils.update_job_record(job_id, {"status": "done", "finished_at": time.time(), "result": result})
    return result
