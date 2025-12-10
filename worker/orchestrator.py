# worker/orchestrator.py

"""
Job Orchestrator with optional LLM integration.

Behavior:
- Uses LLM wrapper if PIBOT_USE_LLM=true; else uses local SymPy solver.
- Supports optional docker-run execution for generated code.
- Persists job status/results in Redis.
- Adds robust safety, JSON extraction and explainer invocation.
"""
from __future__ import annotations

import json
import os
import io
import tarfile
import time
import tempfile
import shutil
import subprocess
import logging
import re
from typing import Any, Dict

import redis
from worker import utils, safety

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
        from worker import llm_wrapper
    except Exception:
        logger.exception("Failed to import llm_wrapper")
        llm_wrapper = None
else:
    llm_wrapper = None


# ---------------------------
# Docker executor
# ---------------------------
def executor_run_docker_script(script_text: str, job_id: str, timeout: int = DOCKER_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    Run `script_text` inside a short-lived container based on EXECUTOR_IMAGE.
    Approach: create container, put_archive (copy) job/job.py into it, exec python /job/job.py,
    capture stdout/stderr/exit_code, remove container.

    Returns dict with keys: stdout, stderr, exit_code, metadata.
    """
    # sanity checks
    if not EXECUTOR_IMAGE:
        return {"stdout": "", "stderr": "no executor image configured", "exit_code": 1, "metadata": {"error": "no_executor_image"}}

    start_ts = time.time()
    try:
        import docker
        from docker.errors import APIError, ContainerError
    except Exception as e:
        return {"stdout": "", "stderr": f"docker-sdk-missing: {e}", "exit_code": 1, "metadata": {"error": str(e)}}

    try:
        client = docker.from_env()
    except Exception as e:
        return {"stdout": "", "stderr": f"docker-client-init-failed: {e}", "exit_code": 1, "metadata": {"error": str(e)}}

    container = None
    stdout = ""
    stderr = ""
    exit_code = 1
    try:
        # 1) create container (not started yet)
        container = client.containers.create(
            image=EXECUTOR_IMAGE,
            command="/bin/sh -c 'sleep 300'",  # keep container alive while we upload + exec
            detach=True,
            tty=False
        )
        container.start()

        # 2) build a tar archive in memory containing job/job.py
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            job_path_in_tar = "job/job.py"  # will create /job/job.py inside container
            data = script_text.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=job_path_in_tar)
            tarinfo.size = len(data)
            tarinfo.mtime = int(time.time())
            tar.addfile(tarinfo, io.BytesIO(data))
        tar_stream.seek(0)

        # 3) put the archive at root '/' so it creates /job/job.py
        success = container.put_archive(path='/', data=tar_stream)
        if not success:
            raise RuntimeError("put_archive failed")

        # 4) exec the job script
        # Use exec_run with demux to get separate stdout/stderr (some docker-py versions)
        try:
            exec_res = container.exec_run(cmd="python /job/job.py", demux=True, stdout=True, stderr=True, tty=False, timeout=timeout)
        except TypeError:
            # older docker SDKs may not accept timeout there; call without timeout and handle externally
            exec_res = container.exec_run(cmd="python /job/job.py", demux=True)

        # exec_res can be (exit_code, (stdout_bytes, stderr_bytes)) when demux=True
        if isinstance(exec_res, tuple) and len(exec_res) == 2 and isinstance(exec_res[1], tuple):
            exit_code = int(exec_res[0]) if exec_res[0] is not None else 0
            out_b, err_b = exec_res[1]
            stdout = (out_b.decode("utf-8", errors="ignore") if out_b else "")
            stderr = (err_b.decode("utf-8", errors="ignore") if err_b else "")
        else:
            # fallback: some versions return an ExecResult-like object
            try:
                exit_code = getattr(exec_res, "exit_code", 0)
                output = getattr(exec_res, "output", None)
                if output is None and hasattr(exec_res, "output"):
                    output = exec_res.output
                if isinstance(output, (bytes, bytearray)):
                    stdout = output.decode("utf-8", errors="ignore")
                else:
                    stdout = str(output)
                stderr = ""
            except Exception:
                stdout = str(exec_res)
                stderr = ""

    except ContainerError as ce:
        # container had non-zero exit
        try:
            if ce.stdout:
                stdout = ce.stdout.decode("utf-8", errors="ignore")
            if ce.stderr:
                stderr = ce.stderr.decode("utf-8", errors="ignore")
        except Exception:
            stderr = str(ce)
        exit_code = getattr(ce, "exit_status", 1)
    except APIError as ae:
        stderr = f"docker-api-error: {ae}"
        exit_code = 1
    except Exception as e:
        stderr = f"executor error: {e}"
        exit_code = 1
    finally:
        # ensure cleanup
        try:
            if container:
                container.remove(force=True)
        except Exception:
            pass
        duration = time.time() - start_ts

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "metadata": {
            "exec_meta": {
                "duration": duration,
            }
        }
    }


# ---------------------------
# Orchestration
# ---------------------------
def orchestrate_job(job_id: str, allow_docker_executor: bool = False) -> Dict[str, Any]:
    """
    Main orchestration entrypoint.
    Steps:
      1) Parse question (LLM parser or heuristic)
      2) Codegen (LLM or sympy_stub)
      3) If llm_codegen -> static safety check -> optionally run in Docker executor
         otherwise fallback to local SymPy solver
      4) Explain (LLM explainer) preferring to pass generated code and executor output
      5) Validate and persist result
    """
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
            logger.debug("LLM parser output: %s", {k: structured.get(k) for k in ("problem_type", "unknowns") if isinstance(structured, dict)})
        except Exception:
            logger.exception("LLM parser failed; falling back to heuristic")
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
    # Step 2: Codegen 
    # ---------------------------
    codegen_out = {"mode": "sympy_stub", "code": None, "metadata": {}}
    if PIBOT_USE_LLM and llm_wrapper:
        try:
            codegen_out = llm_wrapper.generate_python_from_json(structured)
        except Exception as e:
            logger.exception("LLM codegen failed; using sympy_stub: %s", e)
            codegen_out = {"mode": "sympy_stub", "code": None, "metadata": {"error": str(e)}}

    result: Dict[str, Any] = {}

    # If codegen produced LLM code, try to handle execution path
    if codegen_out.get("mode") == "llm_codegen" and codegen_out.get("code"):
        # unwrap code fences if present
        code_text_raw = codegen_out.get("code") or ""
        # prefer llm_wrapper's unwrap if it exists, else use local helper
        if llm_wrapper and hasattr(llm_wrapper, "unwrap_markdown_code"):
            try:
                code_text = llm_wrapper.unwrap_markdown_code(code_text_raw)
            except Exception:
                code_text = utils.unwrap_markdown_code(code_text_raw)
        else:
            code_text = utils.unwrap_markdown_code(code_text_raw)
        # Static Safety check
        safe, issues = False, []
        try:
            safe, issues = safety.is_code_safe(code_text)
        except Exception as e:
            logger.exception("Safety scanner error: %s", e)
            safe, issues = False, [f"safety_scanner_error: {e}"]

        exec_meta: Dict[str, Any] = {
            "requested_solver": requested_solver,
            "codegen_model": codegen_out.get("metadata", {}).get("model"),
            "safety_pass": bool(safe),
            "safety_issues": issues if not safe else [],
            "generated_code_snippet": (code_text[:200] + "...") if code_text else None
        }

        if not safe:
            logger.warning("Code rejected by static safety for job %s: %s", job_id, issues)
            utils.update_job_record(job_id, {
                "status": "failed",
                "finished_at": time.time(),
                "error": {"code": "safety_reject", "message": "Code rejected", "issues": issues},
                "result": {"requested_solver": requested_solver, "solved_by": None, "execution_path": "safety_reject", "metadata": exec_meta}
            })
            return {"error": "safety_reject", "issues": issues}        

         # If Docker executor allowed, run sandboxed
        if allow_docker_executor and DOCKER_RUN_ENABLED:
            exec_out = executor_run_docker_script(code_text, job_id)
            stdout = exec_out.get("stdout", "")
            stderr = exec_out.get("stderr", "")
            exit_code = exec_out.get("exit_code", 1)
            duration = exec_out.get("duration", 0.0)

            exec_meta.update({"stdout_snippet": (stdout[:1000] + "...") if stdout else None,
                              "stderr_snippet": (stderr[:1000] + "...") if stderr else None,
                              "exit_code": exit_code,
                              "duration": duration})
            
            # Try to parse JSON from stdout robustly
            parsed = None
            try:
                parsed = json.loads(stdout) if stdout and stdout.strip().startswith(("{", "[")) else None
            except Exception:
                # fallback to tolerant extractor
                parsed = utils.extract_json_from_text_tolerant(stdout)

            if parsed and isinstance(parsed, dict):
                # ensure required metadata fields
                parsed.setdefault("requested_solver", requested_solver)
                parsed.setdefault("solved_by", "docker_executor")
                parsed.setdefault("execution_path", "executed_in_docker")
                parsed.setdefault("metadata", {}).update({"generated_code": code_text, "codegen_model": codegen_out.get("metadata", {}).get("model"), "exec_meta": exec_meta})
                # If no steps_html present or empty, prefer to call explainer passing code
                if (not parsed.get("steps_html")) and PIBOT_USE_LLM and llm_wrapper:
                    try:
                        expl = llm_wrapper.explain_result(question, parsed, generated_code=code_text)
                        if expl and expl.get("steps_html"):
                            parsed["steps_html"] = expl["steps_html"]
                            parsed.setdefault("explain_meta", {})["from_llm"] = True
                    except Exception:
                        logger.exception("LLM explainer failed for job %s", job_id)
                utils.update_job_record(job_id, {"status": "done", "finished_at": time.time(), "result": parsed})
                return parsed
            else:
                # Executor did not return valid JSON - persist exec metadata and provide fallback or failure
                result = {
                    "requested_solver": requested_solver,
                    "execution_path": "executed_in_docker_no_json",
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "metadata": {"exec_meta": exec_meta}
                }
                utils.update_job_record(job_id, {"status": "failed", "finished_at": time.time(), "error": {"code": "executor_no_json", "message": "Executor did not emit valid JSON on stdout"}, "result": result})
                return result
        else:
            # Docker not available or not allowed — fallback to SymPy stub (but keep code metadata)
            try:
                result = utils.local_sympy_solve_from_question(question)
            except Exception as e:
                utils.update_job_record(job_id, {"status": "failed", "finished_at": time.time(), "error": {"code": "sympy_error", "message": str(e)}})
                raise
            # attach metadata about generated code
            result.setdefault("metadata", {})["generated_code"] = code_text
            result["solved_by"] = "sympy_stub"
            result["execution_path"] = "sympy_fallback_from_llm_codegen"
            result["requested_solver"] = requested_solver
    else:
        # sympy stub path (no LLM codegen)
        try:
            result = utils.local_sympy_solve_from_question(question)
        except Exception as e:
            utils.update_job_record(job_id, {"status": "failed", "finished_at": time.time(), "error": {"code": "sympy_error", "message": str(e)}})
            raise
        result["solved_by"] = "sympy_stub"
        result["execution_path"] = "sympy"
        result["requested_solver"] = requested_solver


    # ---------------------------
    # Step 3 EXPLAIN: prefer LLM explainer if available
    # ---------------------------

    if PIBOT_USE_LLM and llm_wrapper:
        try:
            # pass generated code when available (if we have it in metadata)
            generated_code_for_explain = None
            if isinstance(result.get("metadata"), dict) and result["metadata"].get("generated_code"):
                generated_code_for_explain = result["metadata"].get("generated_code")
            if PIBOT_USE_LLM and llm_wrapper:
                try:
                    expl = llm_wrapper.explain_result(question, result, generated_code=generated_code_for_explain)
                    if expl and expl.get("steps_html"):
                        result["steps_html"] = expl["steps_html"]
                        result.setdefault("explain_meta", {})["from_llm"] = True
                        # record explainer model if available
                        try:
                            result.setdefault("explain_meta", {})["model"] = llm_wrapper.__dict__.get("EXPLAIN_MODEL")
                        except Exception:
                            pass
                except Exception:
                    logger.exception("LLM explainer failed; keeping existing steps_html")
        except Exception:
            logger.exception("Explainer stage error (non-fatal)")

    # ---------------------------
    # Step 4: Validation
    # ---------------------------
    try:
        eqs = structured.get("equations", []) or []
        if eqs and result:
            try:
                eq_objs = utils.build_sympy_eqs(eqs)
                unk = structured.get("unknowns", utils.detect_unknowns(eqs))
                from sympy import symbols
                symb = symbols(" ".join(unk) if unk else "x")
                valid = utils.validate_solution(eq_objs, result.get("structured", {}).get("solutions", result.get("answer")), symb if isinstance(symb, (list, tuple)) else [symb])
                result.setdefault("validation", {})["ok"] = bool(valid)
            except Exception as e:
                logger.debug("Validation skipped/failed: %s", e)
    except Exception:
        logger.exception("Unexpected validation error (non-fatal)")

    # ensure top-level solver metadata exist
    result.setdefault("solved_by", result.get("solved_by", codegen_out.get("mode", "sympy_stub")))
    result.setdefault("requested_solver", requested_solver)

    utils.update_job_record(job_id, {"status": "done", "finished_at": time.time(), "result": result})
    return result

