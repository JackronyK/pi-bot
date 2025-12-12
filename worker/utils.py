# worker/utils.py
"""
Utility Functions - Production-Ready Version
=============================================
Core utilities for the PiBot worker system:
- Redis job management
- Text parsing and normalization
- SymPy equation processing
- Solution validation
- LaTeX/HTML formatting

Features:
- Comprehensive error handling
- Detailed logging
- Type safety with annotations
- Caching for expensive operations
- Batch operations support

Environment Variables:
---------------------
REDIS_URL=redis://redis:6379
PIBOT_CACHE_ENABLED=true|false
PIBOT_LOG_LEVEL=INFO
"""

from __future__ import annotations

import re
import json
import time
import os
import logging
from typing import List, Any, Dict, Tuple, Optional
from functools import lru_cache

import redis
from sympy import symbols, Eq, latex, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.core.sympify import SympifyError

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("worker.utils")
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
CACHE_ENABLED = os.environ.get("PIBOT_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")

# Initialize Redis connection
try:
    redis_conn = redis.from_url(REDIS_URL, decode_responses=True)
    # Test connection
    redis_conn.ping()
    logger.info("✓ Redis connection established")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_conn = None

# ============================================================================
# REDIS JOB MANAGEMENT
# ============================================================================
def get_job_record(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve job record from Redis.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Job dictionary or None if not found
    """
    if not redis_conn:
        logger.error("Redis connection not available")
        return None
    
    key = f"job:{job_id}"
    try:
        raw = redis_conn.get(key)
        if raw:
            job = json.loads(raw)
            logger.debug(f"Retrieved job: {job_id}")
            return job
        else:
            logger.warning(f"Job not found: {job_id}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in job {job_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}")
        return None


def update_job_record(job_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update job record in Redis (shallow merge).
    
    Args:
        job_id: Unique job identifier
        updates: Dictionary of fields to update
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_conn:
        logger.error("Redis connection not available")
        return False
    
    key = f"job:{job_id}"
    try:
        raw = redis_conn.get(key)
        if raw is None:
            logger.warning(f"Cannot update non-existent job: {job_id}")
            return False
        
        job = json.loads(raw)
        job.update(updates)
        redis_conn.set(key, json.dumps(job))
        
        logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating job {job_id}: {e}")
        return False


def create_job_record(job_id: str, job_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
    """
    Create new job record in Redis.
    
    Args:
        job_id: Unique job identifier
        job_data: Job data dictionary
        ttl: Time to live in seconds (optional)
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_conn:
        logger.error("Redis connection not available")
        return False
    
    key = f"job:{job_id}"
    try:
        job_data.setdefault("job_id", job_id)
        job_data.setdefault("created_at", time.time())
        job_data.setdefault("status", "pending")
        
        redis_conn.set(key, json.dumps(job_data))
        
        if ttl:
            redis_conn.expire(key, ttl)
        
        logger.info(f"Created job: {job_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating job {job_id}: {e}")
        return False


def delete_job_record(job_id: str) -> bool:
    """
    Delete job record from Redis.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        True if deleted, False otherwise
    """
    if not redis_conn:
        return False
    
    key = f"job:{job_id}"
    try:
        result = redis_conn.delete(key)
        if result:
            logger.info(f"Deleted job: {job_id}")
        return bool(result)
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        return False
    

# ============================================================================
# TEXT NORMALIZATION & PARSING
# ============================================================================
def strip_leading_instruction(text: str) -> str:
    """
    Remove common instruction prefixes from questions.
    
    Patterns removed:
    - "Solve for x: ..."
    - "Find x: ..."
    - "Compute: ..."
    - "Evaluate: ..."
    - "Simplify: ..."
    
    Args:
        text: Input text
    
    Returns:
        Text with instruction prefix removed
    """
    text = text.strip()
    
    pattern = re.compile(
        r"^\s*(?:solve(?:\s+for\s+[a-zA-Z]\w*)?|"
        r"find(?:\s+for\s+[a-zA-Z]\w*)?|"
        r"compute|evaluate|simplify|calculate)\b[:\s]*(.*)$",
        re.IGNORECASE | re.DOTALL
    )
    
    match = pattern.match(text)
    if match:
        remainder = match.group(1).strip()
        if remainder:
            logger.debug(f"Stripped instruction prefix: {text[:30]}...")
            return remainder
    
    return text


def preprocess_equation_text(text: str) -> str:
    """
    Sanitize equation text for SymPy parsing.
    
    Transformations:
    - Replace ^ with ** (power operator)
    - Remove thousands separators (commas)
    - Normalize whitespace
    
    Args:
        text: Raw equation text
    
    Returns:
        Preprocessed text
    """
    text = text.strip()
    text = text.replace("^", "**")
    text = text.replace(",", "")  # Remove thousands separators
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text


def extract_equations(text: str) -> List[Tuple[str, str]]:
    """
    Extract equations from natural language text.
    
    Supports:
    - Single equations: "x + 5 = 10"
    - Multiple equations: "x + y = 5; 2x - y = 3"
    - System separators: semicolon, newline, pipe
    
    Args:
        text: Text containing equations
    
    Returns:
        List of (lhs, rhs) tuples
    """
    text = strip_leading_instruction(text)
    
    # Split on common separators
    eq_parts = re.split(r"[;\n|]", text)
    equations: List[Tuple[str, str]] = []
    
    for part in eq_parts:
        part = part.strip()
        if "=" in part:
            lhs, rhs = part.split("=", 1)
            equations.append((lhs.strip(), rhs.strip()))
    
    # Fallback: if no equations found but = exists
    if not equations and "=" in text:
        lhs, rhs = text.split("=", 1)
        equations.append((lhs.strip(), rhs.strip()))
    
    logger.debug(f"Extracted {len(equations)} equation(s)")
    return equations

# ============================================================================
# VARIABLE DETECTION
# ============================================================================
def detect_unknowns(equations: List[Tuple[str, str]]) -> List[str]:
    """
    Heuristically detect variable names from equations.
    
    Strategy:
    - Extract single-letter tokens (x, y, z, etc.)
    - Filter out common constants (e, i, pi)
    - Default to ['x'] if none found
    
    Args:
        equations: List of (lhs, rhs) equation tuples
    
    Returns:
        List of variable names
    """
    if not equations:
        return ["x"]
    
    vars_set = set()
    # Constants to exclude
    constants = {"e", "i", "E", "I"}
    
    for lhs, rhs in equations:
        combined = f"{lhs} {rhs}"
        # Find single-letter alphabetic tokens
        tokens = re.findall(r"\b[a-zA-Z]\b", combined)
        for token in tokens:
            if token not in constants:
                vars_set.add(token)
    
    if not vars_set:
        logger.debug("No variables detected, defaulting to ['x']")
        return ["x"]
    
    result = sorted(list(vars_set))
    logger.debug(f"Detected variables: {result}")
    return result



# ============================================================================
# SYMPY EQUATION BUILDING
# ============================================================================
def build_sympy_eqs(equations: List[Tuple[str, str]]) -> List[Eq]:
    """
    Build SymPy Eq objects from string equation tuples.
    
    Features:
    - Implicit multiplication support (2x → 2*x)
    - Standard transformations
    - Detailed error messages
    
    Args:
        equations: List of (lhs_str, rhs_str) tuples
    
    Returns:
        List of SymPy Eq objects
    
    Raises:
        ValueError: If parsing fails
    """
    if not equations:
        raise ValueError("No equations provided")
    
    eq_objects: List[Eq] = []
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    for idx, (lhs_str, rhs_str) in enumerate(equations, 1):
        lhs_text = preprocess_equation_text(lhs_str)
        rhs_text = preprocess_equation_text(rhs_str)
        
        try:
            lhs_expr = parse_expr(lhs_text, transformations=transformations, evaluate=True)
        except (SympifyError, SyntaxError, ValueError) as e:
            raise ValueError(
                f"Failed to parse left side of equation {idx}: '{lhs_str}' → '{lhs_text}'\n"
                f"Error: {e}"
            )
        
        try:
            rhs_expr = parse_expr(rhs_text, transformations=transformations, evaluate=True)
        except (SympifyError, SyntaxError, ValueError) as e:
            raise ValueError(
                f"Failed to parse right side of equation {idx}: '{rhs_str}' → '{rhs_text}'\n"
                f"Error: {e}"
            )
        
        eq_objects.append(Eq(lhs_expr, rhs_expr))
    
    logger.debug(f"Built {len(eq_objects)} SymPy equation(s)")
    return eq_objects


# ============================================================================
# SOLUTION VALIDATION
# ============================================================================
def validate_solution(
    eq_objects: List[Eq],
    solutions: Any,
    unknown_symbols: List[Any],
    tolerance: float = 1e-6
) -> bool:
    """
    Validate solutions by substituting back into equations.
    
    Args:
        eq_objects: List of SymPy equations
        solutions: Solutions to validate (dict, list, or scalar)
        unknown_symbols: List of SymPy symbols or their names
        tolerance: Numerical tolerance for floating point comparison
    
    Returns:
        True if at least one solution satisfies all equations
    """
    try:
        # Normalize unknown symbols to strings
        unknown_names = [str(sym) for sym in unknown_symbols]
        
        # Normalize solutions to list of mappings
        solution_mappings: List[Dict[str, Any]] = []
        
        if isinstance(solutions, dict):
            solution_mappings = [solutions]
        elif isinstance(solutions, (list, tuple, set)):
            if len(unknown_names) == 1:
                # Single variable: list of values
                for val in solutions:
                    if isinstance(val, dict):
                        solution_mappings.append(val)
                    else:
                        solution_mappings.append({unknown_names[0]: val})
            else:
                # Multiple variables: list of dicts
                for sol in solutions:
                    if isinstance(sol, dict):
                        solution_mappings.append(sol)
        else:
            # Single scalar value
            solution_mappings = [{unknown_names[0]: solutions}]
        
        # Validate each solution mapping
        for sol_map in solution_mappings:
            all_satisfied = True
            
            for eq in eq_objects:
                lhs_val = eq.lhs.subs(sol_map)
                rhs_val = eq.rhs.subs(sol_map)
                diff = simplify(lhs_val - rhs_val)
                
                if diff == 0:
                    continue
                
                # Try numerical evaluation
                try:
                    diff_float = float(diff.evalf())
                    if abs(diff_float) < tolerance:
                        continue
                except Exception:
                    pass
                
                all_satisfied = False
                break
            
            if all_satisfied:
                logger.debug("✓ Solution validated successfully")
                return True
        
        logger.debug("✗ No valid solutions found")
        return False
    
    except Exception as e:
        logger.warning(f"Validation error (non-fatal): {e}")
        return False


# ============================================================================
# FORMATTING & DISPLAY
# ============================================================================
def format_steps(
    eq_objects: List[Eq],
    solutions: Any,
    unknown_symbols: List[str]
) -> str:
    """
    Generate HTML with LaTeX math for solution steps.
    
    Args:
        eq_objects: SymPy equations
        solutions: Solution values
        unknown_symbols: Variable names
    
    Returns:
        HTML string with formatted steps
    """
    steps: List[str] = []
    
    # Display equations
    steps.append("<h4>Equation(s):</h4>")
    steps.append("<ol>")
    for eq in eq_objects:
        lhs_latex = latex(eq.lhs)
        rhs_latex = latex(eq.rhs)
        steps.append(f"<li>$$ {lhs_latex} = {rhs_latex} $$</li>")
    steps.append("</ol>")
    
    # Display solutions
    steps.append("<h4>Solution:</h4>")
    
    if isinstance(solutions, dict):
        sol_parts = [f"{k} = {latex(v)}" for k, v in solutions.items()]
        steps.append(f"<p>$$ {', '.join(sol_parts)} $$</p>")
    elif isinstance(solutions, (list, tuple, set)):
        if not solutions:
            steps.append("<p>No solution found.</p>")
        else:
            sol_latex = [latex(s) for s in solutions]
            if len(unknown_symbols) == 1:
                steps.append(
                    f"<p>Solutions for ${unknown_symbols[0]}$: "
                    f"$$ {', '.join(sol_latex)} $$</p>"
                )
            else:
                steps.append(f"<p>Solutions: $$ {', '.join(sol_latex)} $$</p>")
    else:
        steps.append(f"<p>$$ {latex(solutions)} $$</p>")
    
    steps.append("<p><em>Solution verified by substitution.</em></p>")
    
    return "".join(steps)


# ============================================================================
# SOLVER METADATA
# ============================================================================
def normalize_requested_solver_from_job(job: Dict[str, Any]) -> str:
    """
    Extract and normalize requested solver from job record.
    
    Args:
        job: Job dictionary
    
    Returns:
        Normalized solver identifier
    """
    # Check code generation info
    code_info = job.get("simulated_codegen") or job.get("generated_code") or {}
    if isinstance(code_info, dict):
        mode = code_info.get("mode")
        if mode:
            return mode
    
    # Check explicit solver request
    if job.get("requested_solver"):
        return job["requested_solver"]
    
    # Default
    return "sympy_stub"

# ============================================================================
# HIGH-LEVEL SYMPY SOLVER
# ============================================================================
@lru_cache(maxsize=128)
def _cached_sympy_solve(question_hash: str, question: str) -> str:
    """
    Cached version of SymPy solve (internal use only).
    Returns JSON string for caching.
    """
    result = _sympy_solve_impl(question)
    return json.dumps(result)


def _sympy_solve_impl(question: str) -> Dict[str, Any]:
    """
    Internal implementation of SymPy solver.
    
    Args:
        question: Question text
    
    Returns:
        Result dictionary
    
    Raises:
        ValueError: If parsing or solving fails
    """
    # Extract equations
    equations = extract_equations(question)
    
    # Fallback: look for "solve for x: expression"
    if not equations:
        match = re.search(r"solve\s+for\s+([a-zA-Z])[:\s]*(.*)", question, re.IGNORECASE)
        if match:
            var = match.group(1)
            expr = match.group(2).strip()
            if expr:
                equations = [(expr, "0")]
    
    if not equations:
        raise ValueError("Could not extract equations from question")
    
    # Build SymPy equations
    eq_objects = build_sympy_eqs(equations)
    
    # Detect unknowns
    unknown_names = detect_unknowns(equations)
    sympy_symbols = symbols(" ".join(unknown_names))
    if not isinstance(sympy_symbols, (list, tuple)):
        sympy_symbols = [sympy_symbols]
    
    # Solve
    from sympy import solve as sym_solve
    
    try:
        if len(eq_objects) == 1 and len(sympy_symbols) == 1:
            # Single equation, single variable
            sol = sym_solve(eq_objects[0], sympy_symbols[0])
            solutions = sol
        else:
            # System of equations
            sol = sym_solve(eq_objects, sympy_symbols, dict=True)
            solutions = sol
    except Exception as e:
        raise ValueError(f"SymPy solve failed: {e}")
    
    # Validate
    is_valid = validate_solution(eq_objects, solutions, sympy_symbols)
    
    # Format steps
    steps_html = format_steps(eq_objects, solutions, [str(s) for s in sympy_symbols])
    
    # Build result with standardized format
    result = {
        "problem_type": "algebraic equation",
        "raw_question": question,
        "goal": "solve",
        "equations": [{"lhs": lhs, "rhs": rhs} for lhs, rhs in equations],
        "unknowns": unknown_names,
        "solution": {
            "unknown": unknown_names[0] if len(unknown_names) == 1 else "multiple",
            "values": solutions if isinstance(solutions, list) else [solutions]
        },
        "answer": str(solutions),
        "structured": {
            "type": "algebra",
            "solutions": str(solutions)
        },
        "steps_html": steps_html if is_valid else "<p>Solution found but validation failed.</p>",
        "solved_by": "sympy_stub",
        "validation": {
            "ok": bool(is_valid),
            "method": "substitution"
        }
    }
    
    return result

def local_sympy_solve_from_question(question: str) -> Dict[str, Any]:
    """
    Solve mathematical question using SymPy.
    
    Complete pipeline:
    1. Parse equations from text
    2. Build SymPy equation objects
    3. Solve using SymPy
    4. Validate solutions
    5. Format result with HTML/LaTeX
    
    Args:
        question: Natural language question
    
    Returns:
        Dictionary with solution, validation, and formatted output
    
    Raises:
        ValueError: If parsing or solving fails
    """
    logger.info(f"Solving with SymPy: {question[:60]}...")
    
    try:
        if CACHE_ENABLED:
            # Use cached version
            question_hash = str(hash(question))
            result_json = _cached_sympy_solve(question_hash, question)
            result = json.loads(result_json)
        else:
            # Direct solve
            result = _sympy_solve_impl(question)
        
        logger.info("✓ SymPy solve completed")
        return result
    
    except ValueError as e:
        logger.error(f"✗ SymPy solve failed: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error in SymPy solve: {e}", exc_info=True)
        raise ValueError(f"Solver error: {e}")
    

# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================
def unwrap_markdown_code(text: str) -> str:
    """
    Remove markdown code fences from text.
    
    Args:
        text: Text with potential markdown fences
    
    Returns:
        Clean code/text
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Triple backtick fences with language
    match = re.search(r"```(?:python|py|json)?\s*\n(.+?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Inline code fence
    match = re.search(r"```(.+?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Remove standalone fence markers
    text = re.sub(r"^```(?:python|py|json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_json_from_text_tolerant(text: str) -> Optional[Dict[str, Any]]:
    """
    Tolerantly extract JSON from mixed text.
    
    Strategies:
    1. Unwrap markdown fences
    2. Direct JSON parse
    3. Find first {...} block
    4. Progressive truncation
    
    Args:
        text: Text potentially containing JSON
    
    Returns:
        Parsed dictionary or None
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Unwrap markdown
    text = unwrap_markdown_code(text)
    
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find first JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    
    candidate = match.group(1)
    
    # Progressive truncation
    for end_pos in range(len(candidate), 0, -10):
        try:
            return json.loads(candidate[:end_pos])
        except json.JSONDecodeError:
            continue
    
    return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2.5s", "1m 30s")
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
