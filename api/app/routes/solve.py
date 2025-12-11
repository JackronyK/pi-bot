# api/app/routes/solve.py
"""
Solve API Routes
================
Comprehensive API endpoints for the PiBot solving pipeline:
- Question parsing
- Code generation
- Code execution
- Solution explanation
- Full pipeline orchestration
"""

import uuid
import json
import time
import logging
from  typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Worker packages
from worker import llm_wrapper, orchestrator, safety
import redis
from rq import Queue

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter()

# Redis connection for job queue
redis_conn = redis.from_url("redis://redis:6379", decode_responses=True)
rq_queue = Queue("default", connection=redis_conn)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ParseRequest(BaseModel):
    """Request model for question parsing."""
    question: str = Field(..., min_length=1, description="Question text to parse")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Solve for x: x^2 - 7x + 10 = 0"
            }
        }

class CodeGenRequest(BaseModel):
    """Request model for code generation."""
    structured: Dict[str, Any] = Field(..., description="Structured question data")
    
    class Config:
        schema_extra = {
            "example": {
                "structured": {
                    "type": "equation",
                    "equation": "x^2 - 7x + 10 = 0",
                    "variable": "x"
                }
            }
        }


class ExplainRequest(BaseModel):
    """Request model for solution explanation."""
    question: str = Field(..., min_length=1, description="Original question")
    result: Dict[str, Any] = Field(..., description="Execution result")
    generated_code: Optional[str] = Field(None, description="Generated code (optional)")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Solve for x: x^2 - 7x + 10 = 0",
                "result": {"solution": "[2, 5]"},
                "generated_code": "import sympy\nx = sympy.Symbol('x')\n..."
            }
        }


class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""
    generated_code: str = Field(..., min_length=1, description="Python code to execute")
    use_docker_executor: bool = Field(True, description="Use Docker sandbox for execution")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Execution timeout")
    
    @validator('generated_code')
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "generated_code": "import sympy\nx = sympy.Symbol('x')\nresult = sympy.solve(x**2 - 7*x + 10, x)\nprint(result)",
                "use_docker_executor": True,
                "timeout_seconds": 30
            }
        }


class FullPipelineRequest(BaseModel):
    """Request model for full pipeline execution."""
    question: Optional[str] = Field(None, description="Question text")
    structured: Optional[Dict[str, Any]] = Field(None, description="Pre-parsed structure")
    use_docker_executor: bool = Field(True, description="Use Docker sandbox")
    use_llm: bool = Field(True, description="Use LLM wrapper")
    run_async: bool = Field(False, description="Run asynchronously via job queue")
    
    @validator('question', 'structured')
    def validate_question_or_structured(cls, v, values):
        # At least one must be provided
        if 'question' in values and not values.get('question') and not v:
            raise ValueError("Either 'question' or 'structured' must be provided")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Solve for x: x^2 - 7x + 10 = 0",
                "use_docker_executor": True,
                "use_llm": True,
                "run_async": False
            }
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/parse", response_model=Dict[str, Any])
async def parse_question(request: ParseRequest):
    """
    Parse natural language question into structured format.
    
    Args:
        request: ParseRequest containing question text
    
    Returns:
        Dictionary with parsed structure
    
    Raises:
        HTTPException: If parsing fails
    """
    logger.info(f"Parsing question: {request.question[:50]}...")
    
    try:
        parsed_data = llm_wrapper.parse_text_to_json(request.question)
        logger.info("Question parsed successfully")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"parsed": parsed_data, "status": "success"}
        )
    except Exception as e:
        logger.error(f"Parse error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse question: {str(e)}"
        )  
    
# ----- Codegen endpoint -----
@router.post("/codegen", response_model=Dict[str, Any])
async def generate_code(request: CodeGenRequest):
    """
    Generate Python code from structured question data.
    
    Args:
        request: CodeGenRequest containing structured data
    
    Returns:
        Dictionary with generated code and metadata
    
    Raises:
        HTTPException: If code generation fails
    """
    logger.info("Generating code from structured data")
    
    try:
        code_output = llm_wrapper.generate_python_from_json(request.structured)
        logger.info("Code generated successfully")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={**code_output, "status": "success"}
        )
    except Exception as e:
        logger.error(f"Codegen error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate code: {str(e)}"
        )
    
# ----- Explain endpoint -----
@router.post("/explain", response_model=Dict[str, Any])
async def explain_solution(request: ExplainRequest):
    """
    Generate human-readable explanation of solution.
    
    Args:
        request: ExplainRequest with question, result, and optional code
    
    Returns:
        Dictionary with explanation (typically includes steps_html)
    
    Raises:
        HTTPException: If explanation generation fails
    """
    logger.info("Generating explanation")
    
    try:
        explanation = llm_wrapper.explain_result(
            request.question,
            request.result,
            request.generated_code
        )
        logger.info("Explanation generated successfully")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={**explanation, "status": "success"}
        )
    except Exception as e:
        logger.error(f"Explain error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )

 
@router.post("/exec-code", response_model=Dict[str, Any])
async def execute_code(request: ExecuteCodeRequest):
    """
    Execute generated Python code in isolated environment.
    
    Args:
        request: ExecuteCodeRequest with code and execution settings
    
    Returns:
        Dictionary with execution results (stdout, stderr, exit_code, etc.)
    
    Raises:
        HTTPException: If execution fails or code is unsafe
    """
    logger.info("Executing code")
    
    # Perform safety check
    try:
        is_safe, safety_issues = safety.is_code_safe(request.generated_code)
    except Exception as e:
        logger.error(f"Safety check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Safety check error: {str(e)}"
        )
    
    if not is_safe:
        logger.warning(f"Code rejected by safety check: {safety_issues}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "safety_violation",
                "issues": safety_issues,
                "message": "Code contains potentially unsafe operations"
            }
        )
    
    # Execute in Docker if requested
    if request.use_docker_executor:
        job_tag = f"exec-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        logger.info(f"Executing in Docker: {job_tag}")
        
        try:
            execution_output = orchestrator.executor_run_docker_script(
                request.generated_code,
                job_id=job_tag,
                timeout=request.timeout_seconds
            )
            
            # Try to parse JSON output
            try:
                parsed_output = json.loads(execution_output.get("stdout", "{}"))
            except json.JSONDecodeError:
                parsed_output = {
                    "stdout": execution_output.get("stdout", ""),
                    "stderr": execution_output.get("stderr", ""),
                    "exit_code": execution_output.get("exit_code", -1)
                }
            
            parsed_output["execution_path"] = "docker_executor"
            parsed_output["job_tag"] = job_tag
            parsed_output["status"] = "success"
            
            logger.info(f"Execution completed: {job_tag}")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=parsed_output
            )
            
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Code execution failed: {str(e)}"
            )
    else:
        # Docker executor is disabled
        logger.warning("Docker executor disabled - cannot execute code")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Docker executor is required for code execution. Enable it in settings."
        )


@router.post("/full", response_model=Dict[str, Any])
async def run_full_pipeline(request: FullPipelineRequest):
    """
    Execute complete pipeline: parse → codegen → execute → explain.
    
    Can run synchronously (for quick testing) or asynchronously (via job queue).
    
    Args:
        request: FullPipelineRequest with question or structured data
    
    Returns:
        For sync: Complete pipeline results
        For async: Job ID and enqueue status
    
    Raises:
        HTTPException: If pipeline execution fails
    """
    job_id = f"solve_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    logger.info(f"Starting full pipeline: {job_id}")
    
    # Determine structured data
    structured_data = request.structured
    question_text = request.question or ""
    
    if not structured_data:
        if not question_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'question' or 'structured' must be provided"
            )
        
        logger.info("Parsing question as first step")
        try:
            structured_data = llm_wrapper.parse_text_to_json(question_text)
        except Exception as e:
            logger.error(f"Initial parse failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse question: {str(e)}"
            )
    
    # Async execution via job queue
    if request.run_async:
        logger.info(f"Enqueueing async job: {job_id}")
        
        job_record = {
            "job_id": job_id,
            "question": question_text or structured_data.get("raw_question", ""),
            "structured": structured_data,
            "status": "pending",
            "created_at": time.time(),
        }
        
        redis_conn.set(f"job:{job_id}", json.dumps(job_record))
        
        rq_queue.enqueue(
            "worker.orchestrator.orchestrate_job",
            job_id,
            allow_docker_executor=request.use_docker_executor
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "enqueued": True,
                "status": "pending",
                "message": "Job enqueued successfully"
            }
        )
    
    # Synchronous execution
    logger.info(f"Running synchronous pipeline: {job_id}")
    
    try:
        # Create job record for orchestrator
        redis_conn.set(f"job:{job_id}", json.dumps({
            "job_id": job_id,
            "question": question_text or structured_data.get("raw_question", ""),
            "structured": structured_data,
            "status": "running",
            "created_at": time.time(),
        }))
        
        # Run orchestrator
        pipeline_result = orchestrator.orchestrate_job(
            job_id,
            allow_docker_executor=request.use_docker_executor
        )
        
        logger.info(f"Pipeline completed: {job_id}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                **pipeline_result,
                "job_id": job_id,
                "status": "completed"
            }
        )
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Check if solve service is healthy.
    
    Returns:
        Status information
    """
    try:
        # Test Redis connection
        redis_conn.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    
    return JSONResponse(
        status_code=status.HTTP_200_OK if redis_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if redis_ok else "degraded",
            "redis_connected": redis_ok,
            "timestamp": time.time()
        }
    )