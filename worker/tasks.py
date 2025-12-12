# worker/tasks.py
"""
Worker Tasks - Production-Ready Version
========================================
RQ worker task handlers for asynchronous job processing:
- Text problem solving (parse → codegen → execute → explain)
- OCR processing (image/PDF → text extraction)
- File upload handling

Features:
- Comprehensive error handling
- Progress tracking
- Timeout management
- Resource cleanup
- Detailed logging

Environment Variables:
---------------------
REDIS_URL=redis://redis:6379
PIBOT_ENABLE_ORCHESTRATOR=true|false
PIBOT_OCR_DPI=200
PIBOT_OCR_MAX_PAGES=10
"""

from __future__ import annotations

import json
import time
import redis
import os
import logging
import tempfile
import traceback
from  typing import Any, Dict, Optional

from PIL import Image, ImageOps
from pdf2image import convert_from_path
import pytesseract
from worker import utils, safety

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("worker.tasks")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(funcName)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONFIGURATION
# ============================================================================
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
redis_conn = redis.from_url(REDIS_URL, decode_responses=True)

# Feature flags
ENABLE_ORCHESTRATOR = os.environ.get("PIBOT_ENABLE_ORCHESTRATOR", "true").lower() in ("1", "true", "yes")
DOCKER_RUN_ENABLED = os.environ.get("PIBOT_DOCKER_RUN", "true").lower() in ("1", "true", "yes")

# OCR configuration
OCR_DPI = int(os.environ.get("PIBOT_OCR_DPI", "200"))
OCR_MAX_PAGES = int(os.environ.get("PIBOT_OCR_MAX_PAGES", "10"))

# Lazy imports
llm_wrapper = None
orchestrator = None

# ============================================================================
# MODULE IMPORTS (LAZY LOADING)
# ============================================================================
def get_llm_wrapper():
    """Lazy load LLM wrapper module."""
    global llm_wrapper
    if llm_wrapper is None:
        try:
            from worker import llm_wrapper as llm
            llm_wrapper = llm
            logger.info("✓ LLM wrapper loaded")
        except Exception as e:
            logger.warning(f"LLM wrapper unavailable: {e}")
    return llm_wrapper


def get_orchestrator():
    """Lazy load orchestrator module."""
    global orchestrator
    if orchestrator is None:
        try:
            from worker import orchestrator as orch
            orchestrator = orch
            logger.info("✓ Orchestrator loaded")
        except Exception as e:
            logger.warning(f"Orchestrator unavailable: {e}")
    return orchestrator

# ============================================================================
# TEXT PROBLEM SOLVING TASK
# ============================================================================
def process_text_job(job_id: str):
    """
    Process text-based math problem solving job.
    
    Pipeline (new orchestrator path):
    1. Call orchestrator.orchestrate_job()
    2. Update job status
    
    Legacy path (if orchestrator disabled):
    1. Parse question
    2. Generate code
    3. Safety check
    4. Execute (Docker or SymPy fallback)
    5. Generate explanation
    
    Args:
        job_id: Unique job identifier
    """
    logger.info(f"{'='*70}")
    logger.info(f"Processing text job: {job_id}")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    # Load job
    job = utils.get_job_record(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return
    
    question = job.get("question", "")
    if not question:
        logger.error(f"Job {job_id} has no question")
        utils.update_job_record(job_id, {
            "status": "failed",
            "error": {"code": "invalid_job", "message": "No question provided"},
            "finished_at": time.time()
        })
        return
    
    # Mark job as running
    utils.update_job_record(job_id, {
        "status": "running",
        "started_at": start_time
    })

    # ========================================================================
    # NEW PATH: Use Orchestrator (Recommended)
    # ========================================================================
    if ENABLE_ORCHESTRATOR:
        logger.info("Using orchestrator pipeline")

        orch = get_orchestrator()

        if orch:
            try:
                result = orch.orchestrate_job(
                    job_id,
                    allow_docker_executor=DOCKER_RUN_ENABLED
                )
                
                duration = time.time() - start_time
                logger.info(f"✓✓✓ Job completed successfully in {duration:.2f}s")
                return result
            
            except Exception as e:
                logger.error(f"Orchestrator failed: {e}", exc_info=True)
                utils.update_job_record(job_id, {
                    "status": "failed",
                    "error": {
                        "code": "orchestrator_error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    },
                    "finished_at": time.time()
                })
                return
        else:
            logger.warning("Orchestrator not available, falling back to legacy path")

    # ========================================================================
    # LEGACY PATH: Manual Processing
    # ========================================================================
    logger.info("Using legacy processing path")
            
    try:
        _process_text_job_legacy(job_id, job)
    except Exception as e:
        logger.error(f"Legacy processing failed: {e}", exc_info=True)
        utils.update_job_record(job_id, {
            "status": "failed",
            "error": {
                "code": "processing_error",
                "message": str(e),
                "traceback": traceback.format_exc()
            },
            "finished_at": time.time()
        })

def _process_text_job_legacy(job_id: str, job: Dict[str, Any]):
    """
    Legacy processing path (used when orchestrator is disabled).
    
    This maintains backward compatibility with the original task processing.
    """
    requested_solver = utils.normalize_requested_solver_from_job(job)
    logger.info(f"Requested solver: {requested_solver}")
    
    # Get code info
    code_info = job.get("simulated_codegen") or job.get("generated_code") or {}
    code_text = code_info.get("code") if isinstance(code_info, dict) else None
    mode = code_info.get("mode", "") if isinstance(code_info, dict) else ""
    
    # ========================================================================
    # STAGE 1: Safety Check
    # ========================================================================
    if code_text:
        logger.info("Stage 1: Safety validation")
        
        try:
            is_safe, issues = safety.is_code_safe(code_text)
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            is_safe, issues = False, [f"safety_scanner_error: {e}"]
        
        if not is_safe:
            logger.warning(f"Code rejected by safety check: {issues}")
            utils.update_job_record(job_id, {
                "status": "failed",
                "error": {
                    "code": "safety_reject",
                    "message": "Generated code failed safety validation",
                    "issues": issues
                },
                "result": {
                    "requested_solver": requested_solver,
                    "solved_by": None,
                    "execution_path": "safety_reject"
                },
                "finished_at": time.time()
            })
            return
        
        logger.info("✓ Safety check passed")
    
    # ========================================================================
    # STAGE 2: Solve Problem
    # ========================================================================
    logger.info("Stage 2: Solving problem")
    
    result = None
    
    # Try SymPy solver
    try:
        result = utils.local_sympy_solve_from_question(job.get("question", ""))
        result["solved_by"] = "sympy_stub"
        result["execution_path"] = "sympy"
        result["requested_solver"] = requested_solver
        
        if code_text:
            result.setdefault("metadata", {})["generated_code"] = code_text
        
        logger.info("✓ Problem solved with SymPy")
    
    except Exception as e:
        logger.error(f"SymPy solver failed: {e}")
        utils.update_job_record(job_id, {
            "status": "failed",
            "error": {"code": "solver_error", "message": str(e)},
            "finished_at": time.time()
        })
        return
    
    # ========================================================================
    # STAGE 3: Finalize
    # ========================================================================
    utils.update_job_record(job_id, {
        "status": "done",
        "result": result,
        "finished_at": time.time()
    })
    
    logger.info(f"✓ Job completed: {job_id}")

# ============================================================================
# OCR PROCESSING TASKS
# ============================================================================
def save_temp_uploaded_file(upload_bytes: bytes, filename_hint: str = "upload") -> str:
    """
    Save uploaded bytes to temporary file.
    
    Args:
        upload_bytes: File content as bytes
        filename_hint: Prefix for temp file name
    
    Returns:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(prefix=f"{filename_hint}_", suffix="")
    os.close(fd)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(upload_bytes)
        logger.debug(f"Saved temp file: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Failed to save temp file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def image_preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR results.
    
    Steps:
    1. Convert to grayscale
    2. Apply auto-contrast
    3. Resize if too large
    
    Args:
        img: PIL Image
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    img = img.convert("L")
    
    # Auto-contrast
    img = ImageOps.autocontrast(img)
    
    # Resize if too large (improves Tesseract performance)
    max_width = 2000
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized image to {new_size}")
    
    return img


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image: PIL Image
    
    Returns:
        Extracted text
    """
    try:
        # Preprocess
        processed = image_preprocess_for_ocr(image)
        
        # Run OCR
        text = pytesseract.image_to_string(processed)
        logger.debug(f"OCR extracted {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = OCR_DPI,
    max_pages: Optional[int] = OCR_MAX_PAGES
) -> Dict[str, Any]:
    """
    Extract text from PDF using OCR.
    
    Process:
    1. Convert PDF pages to images (using poppler)
    2. Run OCR on each page
    3. Aggregate results
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for PDF rendering
        max_pages: Maximum pages to process (None = all)
    
    Returns:
        Dictionary with page-by-page results and combined text
    """
    logger.info(f"Processing PDF: {pdf_path} (DPI={dpi}, max_pages={max_pages})")
    
    pages_data = []
    
    try:
        # Convert PDF to images
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"PDF converted to {len(pil_pages)} image(s)")
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return {"pages": [], "raw_text": "", "error": str(e)}
    
    # Limit pages if specified
    if max_pages:
        pil_pages = pil_pages[:max_pages]
        if len(pil_pages) < len(pil_pages):
            logger.info(f"Limited to first {max_pages} pages")
    
    # Process each page
    all_text_parts = []
    for idx, pil_image in enumerate(pil_pages, start=1):
        logger.debug(f"Processing page {idx}/{len(pil_pages)}")
        
        text = extract_text_from_image(pil_image)
        pages_data.append({
            "page_no": idx,
            "text": text,
            "char_count": len(text)
        })
        all_text_parts.append(text)
    
    combined_text = "\n\n".join(all_text_parts)
    
    logger.info(
        f"✓ PDF processing complete: {len(pages_data)} pages, "
        f"{len(combined_text)} total characters"
    )
    
    return {
        "pages": pages_data,
        "raw_text": combined_text,
        "page_count": len(pages_data)
    }


def run_math_ocr_placeholder(content: Any) -> Dict[str, Any]:
    """
    Placeholder for specialized math OCR (pix2tex, LaTeX-OCR).
    
    TODO: Integrate actual math OCR model
    
    Args:
        content: Image or text content
    
    Returns:
        Dictionary with LaTeX expressions and confidence
    """
    logger.debug("Math OCR called (placeholder - not yet implemented)")
    return {
        "latex_expressions": [],
        "confidence": 0.0,
        "note": "Math OCR integration pending"
    }


def ocr_task(job_id: str):
    """
    RQ worker entry point for OCR jobs.
    
    Expected job record fields:
    - job_id: Unique identifier
    - file_path: Path to uploaded file
    - file_name: Original filename (optional)
    
    Process:
    1. Load job from Redis
    2. Identify file type (PDF vs image)
    3. Run appropriate OCR
    4. Optionally run math OCR
    5. Store results
    
    Args:
        job_id: Unique job identifier
    """
    logger.info(f"{'='*70}")
    logger.info(f"Processing OCR job: {job_id}")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    # Load job
    job = utils.get_job_record(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return
    
    # Mark as running
    utils.update_job_record(job_id, {
        "status": "running",
        "started_at": start_time
    })
    
    try:
        file_path = job.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"Processing file: {os.path.basename(file_path)} ({file_ext})")
        
        result = {}
        
        # ====================================================================
        # PDF Processing
        # ====================================================================
        if file_ext == ".pdf":
            logger.info("Processing as PDF")
            
            ocr_output = extract_text_from_pdf(
                file_path,
                dpi=OCR_DPI,
                max_pages=OCR_MAX_PAGES
            )
            
            result["pages"] = ocr_output.get("pages", [])
            result["raw_text"] = ocr_output.get("raw_text", "")
            result["page_count"] = ocr_output.get("page_count", 0)
        
        # ====================================================================
        # Image Processing
        # ====================================================================
        else:
            logger.info("Processing as image")
            
            try:
                img = Image.open(file_path)
                text = extract_text_from_image(img)
                
                result["pages"] = [{"page_no": 1, "text": text}]
                result["raw_text"] = text
                result["page_count"] = 1
            
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                raise
        
        # ====================================================================
        # Math OCR (Optional)
        # ====================================================================
        math_ocr_result = run_math_ocr_placeholder(result["raw_text"])
        result["math"] = math_ocr_result
        
        # ====================================================================
        # Finalize
        # ====================================================================
        ocr_result = {
            "job_id": job_id,
            "extracted_text": result["raw_text"],
            "pages": result["pages"],
            "page_count": result.get("page_count", len(result["pages"])),
            "math": result["math"],
            "char_count": len(result["raw_text"]),
            "processing_time": time.time() - start_time
        }
        
        utils.update_job_record(job_id, {
            "status": "done",
            "result": ocr_result,
            "finished_at": time.time()
        })
        
        duration = time.time() - start_time
        logger.info(f"✓ OCR completed successfully in {duration:.2f}s")
        return ocr_result
    
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        
        utils.update_job_record(job_id, {
            "status": "failed",
            "error": {
                "code": "ocr_error",
                "message": str(e),
                "traceback": traceback.format_exc()
            },
            "finished_at": time.time()
        })
        
        return {"error": str(e)}

# ============================================================================
# TASK UTILITIES
# ============================================================================
def cleanup_temp_file(file_path: str):
    """
    Safely remove temporary file.
    
    Args:
        file_path: Path to file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def get_job_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get current progress of a job.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Dictionary with status and progress info
    """
    job = utils.get_job_record(job_id)
    if not job:
        return None
    
    progress = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at")
    }
    
    # Calculate duration if applicable
    if progress["started_at"]:
        if progress["finished_at"]:
            progress["duration"] = progress["finished_at"] - progress["started_at"]
        else:
            progress["duration"] = time.time() - progress["started_at"]
    
    return progress