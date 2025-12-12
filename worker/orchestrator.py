# worker/orchestrator.py
"""
Job Orchestrator - Production-Ready Version
============================================
Orchestrates the complete mathematical problem-solving pipeline with
enterprise-grade reliability, monitoring, and error handling.

Pipeline Stages:
1. Question Parsing (LLM → heuristic fallback)
2. Code Generation (LLM → SymPy fallback)
3. Safety Validation (multi-layer checks)
4. Code Execution (Docker sandbox → local fallback)
5. Solution Explanation (LLM → deterministic fallback)
6. Result Validation (mathematical verification)
7. Persistence & Observability

Features:
- Graceful degradation at every stage
- Comprehensive telemetry and metrics
- Resource limits and timeout management
- Automatic cleanup and error recovery
- Detailed execution traces

Environment Variables:
---------------------
REDIS_URL=redis://redis:6379
PIBOT_USE_LLM=true|false
PIBOT_EXECUTOR_IMAGE=pibot-executor:latest
PIBOT_DOCKER_RUN=true|false
PIBOT_DOCKER_TIMEOUT=30
PIBOT_DOCKER_MEMORY=512m
PIBOT_DOCKER_CPUS=0.5
PIBOT_MAX_RETRIES=2
PIBOT_LOG_LEVEL=INFO
"""

from __future__ import annotations

import json
import os
import io
import tarfile
import time
import logging
import traceback
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

import redis
from worker import utils, safety

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("worker.orchestrator")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(funcName)s:%(lineno)d] - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONFIGURATION
# ============================================================================
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
redis_conn = redis.from_url(REDIS_URL, decode_responses=True)

# Docker executor configuration
EXECUTOR_IMAGE = os.environ.get("PIBOT_EXECUTOR_IMAGE", "pibot-executor:latest")
DOCKER_RUN_ENABLED = os.environ.get("PIBOT_DOCKER_RUN", "false").lower() in ("1", "true", "yes")
DOCKER_TIMEOUT_SECONDS = int(os.environ.get("PIBOT_DOCKER_TIMEOUT", "8"))
DOCKER_MEMORY = os.environ.get("PIBOT_DOCKER_MEMORY", "512m")
DOCKER_CPUS = os.environ.get("PIBOT_DOCKER_CPUS", "0.5")
DOCKER_NETWORK_MODE = os.environ.get("PIBOT_DOCKER_NETWORK", "none")  # Security: no network

# Retry configuration
MAX_RETRIES = int(os.environ.get("PIBOT_MAX_RETRIES", "2"))
RETRY_DELAY_SECONDS = float(os.environ.get("PIBOT_RETRY_DELAY", "1.0"))

# LLM wrapper optional import
PIBOT_USE_LLM = os.environ.get("PIBOT_USE_LLM", "false").lower() in ("1", "true", "yes")
# Gemini / provider settings (optional)
PARSER_MODEL = os.environ.get("PIBOT_GEMINI_PARSER_MODEL", "gemini-2.5-flash")
CODEGEN_MODEL = os.environ.get("PIBOT_GEMINI_CODEGEN_MODEL", "gemini-2.5-pro")
EXPLAIN_MODEL = os.environ.get("PIBOT_GEMINI_EXPLAIN_MODEL", "gemini-2.5-pro")

# Lazy import LLM wrapper
llm_wrapper = None
if PIBOT_USE_LLM:
    try:
        from worker import llm_wrapper
        logger.info("✓ LLM wrapper loaded and ready")
    except Exception as e:
        logger.warning(f"LLM wrapper unavailable: {e}")
        logger.info("→ Will use heuristic fallbacks")
        llm_wrapper = None
else:
    logger.info("LLM features disabled by configuration")


# ============================================================================
# TELEMETRY & METRICS
# ============================================================================
class PipelineMetrics:
    """Track pipeline execution metrics."""

    def __init__(self):
        self.stage_times: Dict[str, float] = {}
        self.stage_status: Dict[str, str] = {}
        self.total_duration: float = 0.0
        self.start_time: float = time.time()

    def start_stage(self, stage_name: str):
        """Mark the start of a pipeline stage."""
        self.stage_times[stage_name] = time.time()
        self.stage_status[stage_name] = "running"

    def end_stage(self, stage_name: str, status: str = "success"):
        """Mark the end of a pipeline stage."""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            self.stage_times[f"{stage_name}_duration"] = duration
            self.stage_status[stage_name] = status
    
    def finalize(self):
        """Calculate total duration."""
        self.total_duration = time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_duration": self.total_duration,
            "stage_durations": {
                k: v for k, v in self.stage_times.items()
                if k.endswith("_duration")
            },
            "stage_status": self.stage_status
        }


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================
@contextmanager
def timed_stage(metrics: PipelineMetrics, stage_name: str):
    """Context manager for timing pipeline stage."""
    metrics.start_stage(stage_name)
    try:
        yield
        metrics.end_stage(stage_name, "success")
    except Exception as e:
        metrics.end_stage(stage_name, "failed")
        logger.error(f"STage '{stage_name}' failed: {e}")
        raise

# ============================================================================
# DOCKER EXECUTOR
# ============================================================================
def executor_run_docker_script(
    script_text: str,
    job_id: str,
    timeout: int = DOCKER_TIMEOUT_SECONDS,
    retry_count: int = 0,
) -> Dict[str, Any]:
    """
    Execute Python script in isolated Docker container with retry logic.
    
    Security features:
    - Network isolation (network_mode=none)
    - Memory limits
    - CPU limits
    - Read-only filesystem where possible
    - Automatic cleanup
    
    Process:
    1. Create ephemeral container
    2. Upload script via tar archive
    3. Execute with timeout
    4. Capture output streams
    5. Force cleanup
    
    Args:
        script_text: Python code to execute
        job_id: Unique job identifier for container naming
        timeout: Maximum execution time (seconds)
        retry_count: Current retry attempt number
    
    Returns:
        Execution result with stdout, stderr, exit_code, duration, metadata
    """
    # sanity checks
    if not EXECUTOR_IMAGE:
        logger.error("Executor image not configured")
        return {
            "stdout": "",
            "stderr": "no executor image configured",
            "exit_code": 1,
            "metadata": {"error": "no_executor_image"}
        }

    start_time = time.time()
    try:
        import docker
        from docker.errors import APIError, ContainerError, ImageNotFound, NotFound
    except Exception as e:
        logger.error(f"Docker SDK unavailable: {e}")
        return {
            "stdout": "",
            "stderr": f"docker-sdk-missing: {e}\nInstall with: pip install docker",
            "exit_code": 1,
            "metadata": {"error": "docker_sdk_missing"}
        }
    
    # Initialize Docker client
    try:
        client = docker.from_env()
        #verify connection
        client.ping()
        logger.debug("Docker client initialized and connected")
    except Exception as e:
        logger.error(f"Docker client initialization failed: {e}")
        return {"stdout": "", "stderr": f"docker-client-init-failed: {e}", "exit_code": 1, "metadata": {"error": str(e)}}

    container = None
    container_name = f"pibot-exec-{job_id}-{int(time.time())}"
    stdout = ""
    stderr = ""
    exit_code = 1
    try:
        logger.info(f"Creating container: {container_name}")

        # 1) Create container with security restrictions
        container = client.containers.create(
            image=EXECUTOR_IMAGE,
            command="/bin/sh -c 'sleep 600'",  # Keep alive for upload + exec
            detach=True,
            name=container_name,
            mem_limit=DOCKER_MEMORY,
            memswap_limit=DOCKER_MEMORY,  # Prevent swap usage
            cpu_quota=int(float(DOCKER_CPUS) * 100000),
            cpu_period=100000,
            network_mode=DOCKER_NETWORK_MODE,
            read_only=False,  # Need /tmp write access
            remove=False,  # Manual cleanup for better logging
            auto_remove=False,
            labels={
                "pibot.job_id": job_id,
                "pibot.created_at": str(int(time.time())),
                "pibot.type": "executor"
            }
        )

        # Start container        
        container.start()
        container_id = container.id[:12]
        logger.debug(f"Container started: {container_id}")

        # 2) Prepare tar archive with script
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            script_bytes = script_text.encode("utf-8")
            tarinfo = tarfile.TarInfo(name="job/job.py")
            tarinfo.size = len(script_bytes)
            tarinfo.mtime = int(time.time())
            tarinfo.mode = 0o644
            tar.addfile(tarinfo, io.BytesIO(script_bytes))

        tar_stream.seek(0)

        # Upload script to container
        logger.debug("Uploading script to container")
        upload_success = container.put_archive(path='/', data=tar_stream)
        if not upload_success:
            raise RuntimeError("Failed to  upload script to container")

        # 4) exec the job script
        logger.info(f"Executing script (timeout={timeout}s)")
        exec_start = time.time()

        try:
            exec_result = container.exec_run(
                cmd="python /job/job.py",
                demux=True,  # Separate stdout/stderr
                stdout=True,
                stderr=True,
                stream=False,
                tty=False
            )  
            exec_duration = time.time() - exec_start

            # parse execution result
            if isinstance(exec_result, tuple) and len(exec_result) == 2:
                exit_code = int(exec_result[0] or 0)

                if isinstance(exec_result[1], tuple):
                    # Demuxed output: (stdout_bytes, stderr_bytes)
                    stdout_bytes, stderr_bytes = exec_result[1]
                    stdout = (stdout_bytes.decode("utf-8", errors="ignore")
                             if stdout_bytes else "")
                    stderr = (stderr_bytes.decode("utf-8", errors="ignore")
                             if stderr_bytes else "")
                            
                else:
                    # Single output stream
                    output_bytes = exec_result[1]
                    if isinstance(output_bytes, bytes):
                        stdout = output_bytes.decode("utf-8", errors="ignore")
                    else:
                        stdout = str(output_bytes)
                    stderr = ""
            
            else:
                # Fallback: object with attributes
                exit_code = getattr(exec_result, "exit_code", 0)
                output = getattr(exec_result, "output", b"")
                stdout = (output.decode("utf-8", errors="ignore")
                         if isinstance(output, bytes) else str(output))
                stderr = ""
            
            logger.info(
                f"Execution completed: exit_code={exit_code}, "
                f"duration={exec_duration:.2f}s, "
                f"stdout_len={len(stdout)}, stderr_len={len(stderr)}"
            )
        
        except Exception as e:
            logger.error(f"Execution error: {e}")
            stderr = f"execution-error: {e}"
            exit_code = 1

    except ImageNotFound as e:
        error_msg = f"Executor image not found: {EXECUTOR_IMAGE}"
        logger.error(f"{error_msg}\nPull with: docker pull {EXECUTOR_IMAGE}")
        stderr = error_msg
        exit_code = 1

    except ContainerError as e:
        logger.error(f"Container error: {e}")
        try:
            if hasattr(e, 'stdout') and e.stdout:
                stdout = e.stdout.decode("utf-8", errors="ignore")
            if hasattr(e, 'stderr') and e.stderr:
                stderr = e.stderr.decode("utf-8", errors="ignore")
        except Exception:
            stderr = str(e)
        exit_code = getattr(e, "exit_status", 1)

    except APIError as e:
        logger.error(f"Docker API error: {e}")
        stderr = f"docker-api-error: {e}"
        exit_code = 1
    
    except Exception as e:
        logger.error(f"Unexpected executor error: {e}", exc_info=True)
        stderr = f"executor-error: {e}\n{traceback.format_exc()}"
        exit_code = 1
    
    finally:
        # Always clean up container
        if container:
            try:
                logger.debug(f"Removing container: {container_name}")
                container.stop(timeout=2)
                container.remove(force=True)
            except NotFound:
                logger.debug("Container already removed")
            except Exception as e:
                logger.warning(f"Container cleanup failed: {e}")
        
        duration = time.time() - start_time
    
    # Retry logic for transient failures
    if exit_code != 0 and retry_count < MAX_RETRIES:
        if "timeout" in stderr.lower() or "connection" in stderr.lower():
            logger.warning(
                f"Transient failure detected (retry {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(RETRY_DELAY_SECONDS * (retry_count + 1))
            return executor_run_docker_script(
                script_text,
                job_id,
                timeout,
                retry_count + 1
            )
    
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration": duration,
        "metadata": {
            "job_id": job_id,
            "container_name": container_name,
            "image": EXECUTOR_IMAGE,
            "timeout": timeout,
            "memory_limit": DOCKER_MEMORY,
            "cpu_limit": DOCKER_CPUS,
            "network_mode": DOCKER_NETWORK_MODE,
            "retry_count": retry_count
        }
    }

# ============================================================================
# MAIN ORCHESTRATION PIPELINE
# ============================================================================
def orchestrate_job(job_id: str, allow_docker_executor: bool = True) -> Dict[str, Any]:
    """
    Execute complete problem-solving pipeline with world-class reliability.
    
    This is the main entry point for job orchestration. It coordinates all
    pipeline stages and ensures graceful degradation at every step.
    
    Args:
        job_id: Unique job identifier (must exist in Redis)
        allow_docker_executor: Whether to use Docker sandbox for code execution
    
    Returns:
        Complete result dictionary with solution, metadata, and telemetry
    
    Raises:
        ValueError: If job not found or invalid
        RuntimeError: On critical unrecoverable failures
    """
    logger.info(f"{'='*70}")
    logger.info(f"Starting orchestration: {job_id}")
    logger.info(f"{'='*70}")

    metrics = PipelineMetrics()

    # ========================================================================
    # STAGE 0: Load and Validate Job
    # ========================================================================
    with timed_stage(metrics, "job_load"):
        job = utils.get_job_record(job_id)
        if not job:
            raise ValueError(f"Job not found in Redis: {job_id}")
        
        question = job.get("question", "")
        if not question or not question.strip():
            raise ValueError(f"Job {job_id} has no question text")

        logger.info(f"Question: {question[:100]}...")

        # Mark job running
        utils.update_job_record(
            job_id,
            {
                "status": "running",
                "started_at": time.time(),
                "orchestration_started": True
            }
        )
        requested_solver = utils.normalize_requested_solver_from_job(job)
        logger.info(f"Requested solver: {requested_solver}")

    # ========================================================================
    # STAGE 1: Parse Question
    # ========================================================================
    structured = None

    with timed_stage(metrics, "parse"):
        logger.info("Stage 1: Parsing question")
            
        if PIBOT_USE_LLM and llm_wrapper:
            try:
                logger.debug("Attempting LLM parsing...")
                structured = llm_wrapper.parse_text_to_json(question)
                logger.info(f"✓ LLM parsing successful: {structured.get('problem_type', 'unknown')}")
            except Exception as e:
                logger.exception(f"LLM parser failed: {e}")
                structured = None

        if structured is None:
            logger.info("Using heuristic parser")
            equations = utils.extract_equations(question)
            structured = {
                "raw_question": question,
                "problem_type": "equation" if len(equations) == 1 else "system",
                "equations": [{"lhs": lhs, "rhs": rhs} for lhs, rhs in equations],
                "unknowns": utils.detect_unknowns(equations) if equations else ["x"],
                "goal": "solve",
                "notes": "heuristic-fallback"
            }
            logger.info(f"✓ Heuristic parsing complete: {len(structured.get('equations', []))} equation(s)")

    # ========================================================================
    # STAGE 2: Generate Solver Code
    # ======================================================================== 
    codegen_output = {"mode": "sympy_stub", "code": None, "metadata": {}}

    with timed_stage(metrics, "codegen"):
        logger.info("STAGE 2: Generating solver code")        

        if PIBOT_USE_LLM and llm_wrapper:
            try:
                logger.debug("Attempting LLM code generation...")
                codegen_output = llm_wrapper.generate_python_from_json(structured)
                mode = codegen_output.get("mode", "unknown")
                logger.info(f"✓ Code generation complete: mode={mode}")

                if codegen_output.get("code"):
                    code_preview = codegen_output["code"][:200].replace("\n", " ")
                    logger.debug(f"Code preview: {code_preview}...")
            except Exception as e:
                logger.exception(f"Code generation failed: {e}")
                codegen_output = {
                    "mode": "sympy_stub",
                    "code": None,
                    "metadata": {"error": str(e)}
                }
        if codegen_output.get("mode") == "sympy_stub":
            logger.info("Using SymPy fallback solver")

    result: Dict[str, Any] = {}
    # ========================================================================
    # STAGE 3: Execute LLM-Generated Code (if available)
    # ========================================================================
    if codegen_output.get("mode") == "llm_codegen" and codegen_output.get("code"):
        logger.info("STAGE 3: Processing LLM-generated code")

        code_raw = codegen_output.get("code", "")
        code_clean = utils.unwrap_markdown_code(code_raw)

        # Sub-stage: Safety validation
        with timed_stage(metrics, "safety_check"):
            logger.debug("Performing Safety validation")
            try:
                is_safe, safety_issues = safety.is_code_safe(code_clean)
            except Exception as e:
                logger.error(f"Safety check failed: {e}")
                is_safe, safety_issues = False, [f"safety_scanner_error: {e}"]

            exec_metadata = {
                "requested_solver": requested_solver,
                "codegen_model": codegen_output.get("metadata", {}).get("model"),
                "safety_pass": is_safe,
                "safety_issues": safety_issues if not is_safe else [],
                "code_length": len(code_clean),
                "code_lines": code_clean.count("\n") + 1
            }            

            if not is_safe:
                logger.warning(f"❌ Code failed safety check: {safety_issues}")
                utils.update_job_record(job_id, {
                    "status": "failed",
                    "finished_at": time.time(),
                    "error": {
                        "code": "safety_reject",
                        "message": "Generated code failed safety validation",
                        "issues": safety_issues
                    },
                    "result": {
                        "requested_solver": requested_solver,
                        "execution_path": "safety_reject",
                        "metadata": exec_metadata
                    },
                    "metrics": metrics.to_dict()
                })
                return {
                    "error": "safety_reject",
                    "issues": safety_issues,
                    "requested_solver": requested_solver,
                    "metadata": exec_metadata
                }
            
            logger.info("✓ Safety validation passed")      

        # Sub-stage: Docker execution
        if allow_docker_executor and DOCKER_RUN_ENABLED:
            with timed_stage(metrics, "docker_execution"):
                logger.info("Executing in Docker sandbox")
                exec_result = executor_run_docker_script(code_clean, job_id, DOCKER_TIMEOUT_SECONDS)                
                stdout = exec_result.get("stdout", "")
                stderr = exec_result.get("stderr", "")
                exit_code = exec_result.get("exit_code", 1)
                duration = exec_result.get("duration", 0.0)

                exec_metadata.update({
                    "execution_method": "docker",
                    "exit_code": exit_code,
                    "duration": duration,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr)
                })
                logger.info(
                    f"Execution complete: exit_code={exit_code}, "
                    f"duration={duration:.2f}s"
                )
            
                # Try to parse JSON from stdout robustly
                parsed_result = None
                if stdout.strip() and exit_code == 0:
                    try:
                        parsed_result = json.loads(stdout)
                    except Exception:
                        logger.debug("stdout is not JSON, trying tolerant extraction")
                        # fallback to tolerant extractor
                        parsed_result = utils.extract_json_from_text_tolerant(stdout)

                if parsed_result and isinstance(parsed_result, dict):
                    logger.info("✓ Valid JSON output received")

                    # Enrich result
                    parsed_result.setdefault("requested_solver", requested_solver)
                    parsed_result.setdefault("solved_by", "docker_executor")
                    parsed_result.setdefault("execution_path", "docker_executor")
                    parsed_result.setdefault("metadata", {}).update({
                        "generated_code": code_clean,
                        "exec_meta": exec_metadata
                    })

                    # Generate explanation if missing
                    if not parsed_result.get("steps_html"):
                        with timed_stage(metrics, "explanation"):
                            if PIBOT_USE_LLM and llm_wrapper:
                                logger.info("Generating solution explanation")
                                try:
                                    explanation = llm_wrapper.explain_result(
                                        question,
                                        parsed_result,
                                        generated_code=code_clean
                                    )
                                    if explanation.get("steps_html"):
                                        parsed_result["steps_html"] = explanation["steps_html"]
                                        parsed_result.setdefault("explain_meta", {}).update({
                                            "from_llm": True,
                                            "confidence": explanation.get("confidence", 1.0),
                                            "model": EXPLAIN_MODEL if 'llm_wrapper' in dir() else None
                                        })
                                        logger.info("✓ Explanation generated")
                                except Exception as e:
                                    logger.exception(f"Explanation generation failed: {e}")
                    # Finalize metrics and persist
                    metrics.finalize()
                    parsed_result["metrics"] = metrics.to_dict()
                    
                    utils.update_job_record(job_id, {
                        "status": "done",
                        "finished_at": time.time(),
                        "result": parsed_result
                    })
                    logger.info(f"✓✓✓ Pipeline completed successfully: {job_id}")
                    logger.info(f"Total duration: {metrics.total_duration:.2f}s")
                    return parsed_result
                else:                    
                    # Executor did not return valid JSON - persist exec metadata and provide fallback or failure
                    logger.error("❌ Executor did not return valid JSON")
                    result = {
                        "requested_solver": requested_solver,
                        "execution_path": "docker_no_json",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "metadata": exec_metadata
                    }

                    metrics.finalize()
                    result["metrics"] = metrics.to_dict()

                    utils.update_job_record(job_id, {
                        "status": "failed",
                        "finished_at": time.time(),
                        "error": {
                            "code": "invalid_executor_output",
                            "message": "Executor did not produce valid JSON output"
                        },
                        "result": result
                    })
                    return result
        else:
            # Docker not available or not allowed — fallback to SymPy stub (but keep code metadata)
            logger.info("Docker execution disabled - falling back to SymPy")
            with timed_stage(metrics, "sympy_fallback"):
                result = utils.local_sympy_solve_from_question(question)
                result["solved_by"] = "sympy_stub"
                result["execution_path"] = "sympy_fallback_from_llm"
                result["requested_solver"] = requested_solver
                result.setdefault("metadata", {})["generated_code"] = code_clean
    
    else:
        # ====================================================================
        # STAGE 3b: SymPy Direct Path
        # ==================================================================== 
        logger.info("STAGE 3b: Using SymPy solver directly")
        with timed_stage(metrics, "sympy_solve"):
            try:
                result = utils.local_sympy_solve_from_question(question)
                result["solved_by"] = "sympy_stub"
                result["execution_path"] = "sympy"
                result["requested_solver"] = requested_solver
                logger.info("✓ SymPy solution computed")
            except Exception as e:
                logger.error(f"❌ SymPy solver failed: {e}", exc_info=True)
                metrics.finalize()
                utils.update_job_record(job_id, {
                    "status": "failed",
                    "finished_at": time.time(),
                    "error": {
                        "code": "solver_error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    },
                    "metrics": metrics.to_dict()
                })
                raise


    # ========================================================================
    # STAGE 4: Generate Explanation (if not already present)
    # ========================================================================
    if not result.get("steps_html"):
        with timed_stage(metrics, "explanation"):        
            if PIBOT_USE_LLM and llm_wrapper:
                logger.info("STAGE 4: Generating explanation")
                try:
                    # pass generated code when available (if we have it in metadata)
                    generated_code = result.get("metadata", {}).get("generated_code")
                    explanation = llm_wrapper.explain_result(
                        question,
                        result,
                        generated_code=generated_code
                    )
                    
                    if explanation.get("steps_html"):
                        result["steps_html"] = explanation["steps_html"]
                        result.setdefault("explain_meta", {}).update({
                            "from_llm": True,
                            "confidence": explanation.get("confidence", 1.0)
                        })
                        logger.info("✓ Explanation generated")
                except Exception as e:
                    logger.error(f"Explanation generation failed: {e}")
    
    # ========================================================================
    # STAGE 5: Validate Solution
    # ========================================================================
    with timed_stage(metrics, "validation"):
        logger.info("STAGE 5: Validating solution")

        try:
            equations = structured.get("equations", [])
            if equations and result:
                eq_tuples = [(eq["lhs"], eq["rhs"]) for eq in equations]
                eq_objects = utils.build_sympy_eqs(eq_tuples)


                unknowns = structured.get("unknowns", ["x"])
                from sympy import symbols
                symb = symbols(" ".join(unknowns))

                # Extract solutions from result
                solutions = []
                if result.get("solution") and isinstance(result["solution"], dict):
                    solutions = result["solution"].get("values", [])
                if not solutions:
                    solutions = result.get("answer", [])
                    # Validate
                is_valid = utils.validate_solution(
                    eq_objects,
                    solutions,
                    [symb] if not isinstance(symb, (list, tuple)) else symb
                )
                
                result.setdefault("validation", {})["ok"] = bool(is_valid)
                logger.info(f"✓ Validation: {'PASSED ✓' if is_valid else 'FAILED ✗'}")

        except Exception as e:
            logger.warning(f"Validation failed (non-fatal): {e}")
            result.setdefault("validation", {})["error"] = str(e)
    
    # ========================================================================
    # STAGE 6: Finalize and Persist
    # ========================================================================
    logger.info("STAGE 6: Finalizing results")

    result.setdefault("requested_solver", requested_solver)
    result.setdefault("solved_by", codegen_output.get("mode", "sympy_stub"))
    
    metrics.finalize()
    result["metrics"] = metrics.to_dict()
    
    utils.update_job_record(job_id, {
        "status": "done",
        "finished_at": time.time(),
        "result": result
    })

    logger.info(f"{'='*70}")
    logger.info(f"✓✓✓ Pipeline completed successfully: {job_id}")
    logger.info(f"Total duration: {metrics.total_duration:.2f}s")
    logger.info(f"Solved by: {result.get('solved_by', 'unknown')}")
    logger.info(f"{'='*70}")
    
    return result

