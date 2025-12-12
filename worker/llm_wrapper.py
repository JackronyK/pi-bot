# worker/llm_wrapper.py
"""
LLM Wrapper - Production-Ready Version
=======================================
Provides AI-powered mathematical problem-solving capabilities using Google Gemini.

Core Functions:
- Question Parsing: Natural language → Structured JSON
- Code Generation: Structured data → Executable Python
- Solution Explanation: Results → Human-readable HTML

Features:
- Multi-strategy JSON extraction
- Robust error handling with fallbacks
- Automatic markdown fence removal
- Safety validation for generated code
- Retry logic with exponential backoff
- Comprehensive logging and telemetry

Environment Variables:
---------------------
GEMINI_API_KEY=*****        

PIBOT_USE_LLM=true|false
PIBOT_LLM_PROVIDER=gemini|mock

PIBOT_GEMINI_PARSER_MODEL=gemini-2.0-flash-exp
PIBOT_GEMINI_CODEGEN_MODEL=gemini-2.0-flash-exp  
PIBOT_GEMINI_EXPLAIN_MODEL=gemini-2.0-flash-exp

PIBOT_GEMINI_PARSER_TEMP=0.0
PIBOT_GEMINI_CODEGEN_TEMP=0.0
PIBOT_GEMINI_EXPLAIN_TEMP=0.2

PIBOT_MAX_RETRIES=2
PIBOT_RETRY_DELAY=1.0
"""
from __future__ import annotations

import os
import json
import re
import logging
import time
from typing import Any, Dict, Optional,Tuple, List

# local helpers
from worker import utils

# genai client
try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("worker.llm_wrapper")
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
PIBOT_USE_LLM = os.environ.get("PIBOT_USE_LLM", "false").lower() in ("1", "true", "yes")
PIBOT_LLM_PROVIDER = os.environ.get("PIBOT_LLM_PROVIDER", "gemini").lower()  # 'gemini' or 'mock'
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Model Selection
PARSER_MODEL = os.environ.get("PIBOT_GEMINI_PARSER_MODEL", "gemini-2.5-flash")
CODEGEN_MODEL = os.environ.get("PIBOT_GEMINI_CODEGEN_MODEL", "gemini-2.5-pro")
EXPLAIN_MODEL = os.environ.get("PIBOT_GEMINI_EXPLAIN_MODEL", "gemini-2.5-pro")

# LLM runtime params
PARSER_TEMP = float(os.environ.get("PIBOT_GEMINI_PARSER_TEMP", "0.0"))
CODEGEN_TEMP = float(os.environ.get("PIBOT_GEMINI_CODEGEN_TEMP", "0.0"))
EXPLAIN_TEMP = float(os.environ.get("PIBOT_GEMINI_EXPLAIN_TEMP", "0.2"))

# Retry configuration
MAX_RETRIES = int(os.environ.get("PIBOT_MAX_RETRIES", "2"))
RETRY_DELAY = float(os.environ.get("PIBOT_RETRY_DELAY", "1.0"))

FORBIDDEN_IMPORTS = re.compile(
    r"\b(import|from)\s+(os|sys|subprocess|socket|shutil|ctypes|"
    r"multiprocessing|threading|requests|urllib|http|pickle|"
    r"eval|exec|compile|__import__)\b",
    flags=re.IGNORECASE
)

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================
client: Optional[genai.Client] = None

if GENAI_AVAILABLE and PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini":
    try:
        client = genai.Client()
        logger.info("✓ Gemini client initialized successfully")
        logger.info(f"  Parser: {PARSER_MODEL}")
        logger.info(f"  Codegen: {CODEGEN_MODEL}")
        logger.info(f"  Explainer: {EXPLAIN_MODEL}")
    except Exception:
        logger.error(f"Failed to initialize Gemini client: {e}")
        logger.warning("→ LLM features will use fallback methods")
        client = None
else:
    if not GENAI_AVAILABLE and PIBOT_USE_LLM:
        logger.warning("google-genai not installed. Install: pip install google-genai")
    if not PIBOT_USE_LLM:
        logger.info("LLM features disabled by configuration")
    client = None 

# ============================================================================
# PROMPT MANAGEMENT
# ============================================================================
def _read_prompt(name: str) -> str:
    """
    Load prompt template from prompts directory.
    
    Args:
        name: Prompt filename (e.g., "parser.md")
    
    Returns:
        Prompt content or empty string if not found
    """
    path = os.path.join(PROMPT_DIR, name)
    try:
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
            logger.debug(f"Loaded prompt: {name} ({len(content)} chars)")
            return content
    except FileNotFoundError:
        logger.debug("Prompt file not found: %s", path)
        return ""
    except Exception as e:
        logger.exception("Error reading prompt file %s: %s", path, e)
        return ""


# ============================================================================
# GEMINI API WRAPPER
# ============================================================================
def _call_gemini(prompt: str, model: str, temperature: float, max_retries: int = MAX_RETRIES) -> str:
    """
    Call Gemini API with retry logic and comprehensive error handling.
    
    Features:
    - Exponential backoff on transient failures
    - Multiple response format support
    - Detailed logging
    
    Args:
        prompt: Input prompt text
        model: Gemini model identifier
        temperature: Sampling temperature (0.0 = deterministic)
        max_retries: Maximum retry attempts
    
    Returns:
        Generated text response
    
    Raises:
        RuntimeError: If client unavailable or all retries exhausted
    """
    if client is None:
        raise RuntimeError(
            "Gemini client not initialized. "
            "Set GEMINI_API_KEY environment variable and ensure google-genai is installed."
        )
    
    logger.debug(
        f"Calling Gemini: model={model}, temp={temperature}, "
        f"prompt_len={len(prompt)}, retries={max_retries}"
    )

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            
            # Make API call
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            
            duration = time.time() - start_time

            # Extract text from response (handle multiple formats)
            if hasattr(response, "text") and response.text:
                result = response.text
                logger.debug(
                    f"✓ Gemini response received: {len(result)} chars "
                    f"in {duration:.2f}s"
                )                
                return result
            # Handle candidates structure
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content"):
                    content = candidate.content
                    if hasattr(content, "parts") and content.parts:
                        text_parts = [
                            part.text for part in content.parts
                            if hasattr(part, "text") and part.text
                        ]
                        if text_parts:
                            result = "".join(text_parts)
                            logger.debug(
                                f"✓ Extracted from parts: {len(result)} chars "
                                f"in {duration:.2f}s"
                            )
                            return result
            # Fallback to string representation
            result = str(response)
            logger.warning(
                f"Using fallback string representation: {result[:100]}..."
            )
            return result
        except Exception as e:
            last_error = e
            
            if attempt < max_retries:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Gemini call failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}"
                )
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Gemini call failed after {max_retries + 1} attempts: "
                    f"{type(e).__name__}: {e}"
                )
    
    raise RuntimeError(f"Gemini API call failed: {last_error}")


# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text using multiple strategies.
    
    Strategies (in order):
    1. Direct JSON parsing
    2. Remove markdown fences and parse
    3. Find first complete JSON object via brace matching
    4. Fix common issues (trailing commas, comments)
    5. Progressive truncation
    
    Args:
        text: Text potentially containing JSON
    
    Returns:
        Parsed dictionary or None
    """
    if not text or not isinstance(text, str):
        return None
    
    # Strategy 1: Remove markdown fences
    cleaned = re.sub(r"```(?:json|js|javascript)?\s*\n?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" `\n\r\t")
    
    # Strategy 2: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find first complete JSON object
    match = re.search(r"\{", cleaned)
    if not match:
        return None
    
    start_idx = match.start()
    depth = 0
    end_idx = None
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\":
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if end_idx is None:
        return None
    
    json_candidate = cleaned[start_idx:end_idx]
    
    # Strategy 4: Try parsing
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Fix common issues
    fixed = json_candidate
    # Remove trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    # Remove single-line comments
    fixed = re.sub(r'//[^\n]*\n', '\n', fixed)
    # Remove multi-line comments
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Strategy 6: Progressive truncation (last resort)
    for end in range(len(json_candidate), start_idx, -10):
        try:
            return json.loads(json_candidate[:end])
        except json.JSONDecodeError:
            continue
    
    logger.warning(f"Failed to extract JSON from text ({len(text)} chars)")
    return None


# ============================================================================
# CODE UTILITIES
# ============================================================================
def unwrap_markdown_code(text: str) -> str:
    """
    Remove markdown code fences from text.
    
    Handles:
    - Triple backtick fences with optional language
    - Inline code fences
    - Leading/trailing commentary
    
    Args:
        text: Text potentially containing fenced code
    
    Returns:
        Clean code string
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Match fenced code block with language specifier
    match = re.search(
        r"```(?:python|py|code)?\s*\n(.+?)\n```",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    
    # Match inline code fence
    match = re.search(r"```(.+?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Remove standalone fence markers
    text = re.sub(r"^```(?:python|py|code)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def check_code_safety_basic(code: str) -> Tuple[bool, List[str]]:
    """
    Perform basic safety pre-check on generated code.
    
    Note: This is a lightweight check. Full safety scanning
    happens later via the safety module.
    
    Args:
        code: Python code to check
    
    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    issues = []
    
    if not code or not code.strip():
        issues.append("Code is empty")
        return False, issues
    
    # Check for forbidden imports
    forbidden_matches = FORBIDDEN_IMPORTS.findall(code)
    if forbidden_matches:
        unique_imports = set(match[1] for match in forbidden_matches)
        issues.append(f"Forbidden imports: {', '.join(unique_imports)}")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        (r"\beval\s*\(", "eval() function"),
        (r"\bexec\s*\(", "exec() function"),
        (r"\b__import__\s*\(", "__import__() function"),
        (r"\bcompile\s*\(", "compile() function"),
        (r"open\s*\([^)]*['\"]w", "file write operation"),
        (r"\bshutil\.", "shutil module usage"),
        (r"\bos\.system", "os.system() call"),
        (r"\bsubprocess\.", "subprocess module usage"),
    ]
    
    for pattern, description in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(f"Suspicious pattern: {description}")
    
    is_safe = len(issues) == 0
    
    if not is_safe:
        logger.warning(f"Basic safety check found {len(issues)} issue(s)")
    
    return is_safe, issues

# ============================================================================
# PUBLIC API
# ============================================================================
def parse_text_to_json(question: str, *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse natural language question into structured JSON.
    
    Pipeline:
    1. Try LLM parsing (if available)
    2. Fallback to heuristic extraction
    3. Normalize and validate structure
    
    Args:
        question: Natural language math question
        model_cfg: Optional model configuration overrides
    
    Returns:
        Structured dictionary with:
        - problem_type: Type classification
        - raw_question: Original text
        - goal: What to accomplish
        - equations: List of equation objects
        - unknowns: List of variables
        - notes: Additional metadata
    """
    logger.info(f"Parsing question ({len(question)} chars)")

    prompt_template = _read_prompt("parser.md")
    if not prompt_template:
        prompt_template = """Parse the following mathematical question into structured JSON.

Return a JSON object with these fields:
- problem_type: "equation" | "system" | "inequality" | "expression"
- raw_question: original question text
- goal: "solve" | "simplify" | "evaluate"
- equations: array of {lhs: "...", rhs: "..."}
- unknowns: array of variable names

Return ONLY the JSON object, no explanation.

Example:
{
  "problem_type": "equation",
  "raw_question": "Solve for x: 2x + 5 = 13",
  "goal": "solve",
  "equations": [{"lhs": "2*x + 5", "rhs": "13"}],
  "unknowns": ["x"]
}"""
    full_prompt = f"{prompt_template}\n\n# QUESTION:\n{question}\n\n# JSON:\n"

    # Try LLM parsing
    if PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini" and client:
        try:
            logger.debug("Attempting LLM parsing...")
            response_text = _call_gemini(
                full_prompt,
                model=PARSER_MODEL,
                temperature=PARSER_TEMP
            )
            
            parsed = _extract_json_from_text(response_text)

            if parsed and isinstance(parsed, dict):
                # Normalize
                parsed = normalize_parsed_structure(parsed, question)
                logger.info(f"✓ LLM parsing successful: {parsed.get('problem_type')}")
                return parsed
            else:
                logger.warning("LLM returned invalid JSON, using fallback")

        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")            

    # Heuristic fallback
    logger.info("Using heuristic parser")
    return heuristic_parse(question)

def heuristic_parse(question: str) -> Dict[str, Any]:
    """
    Fallback heuristic parser using regex and pattern matching.
    
    Args:
        question: Question text
    
    Returns:
        Basic structured representation
    """
    try:
        equations = utils.extract_equations(question)
        unknowns = utils.detect_unknowns(equations) if equations else ["x"]
        
        equations_normalized = [
            {"lhs": lhs, "rhs": rhs}
            for lhs, rhs in equations
        ]
        
        problem_type = "equation" if len(equations) == 1 else "system"
        
        logger.debug(
            f"Heuristic parse: {problem_type}, "
            f"{len(equations_normalized)} equation(s), "
            f"unknowns={unknowns}"
        )
        
        return {
            "problem_type": problem_type,
            "raw_question": question,
            "goal": "solve",
            "equations": equations_normalized,
            "unknowns": unknowns,
            "notes": "heuristic-fallback"
        }
    
    except Exception as e:
        logger.error(f"Heuristic parsing failed: {e}")
        return {
            "problem_type": "unknown",
            "raw_question": question,
            "goal": "solve",
            "equations": [],
            "unknowns": ["x"],
            "notes": f"error-fallback: {str(e)}"
        }

def normalize_parsed_structure(
    parsed: Dict[str, Any],
    original_question: str
) -> Dict[str, Any]:
    """
    Normalize and validate LLM-parsed structure.
    
    Ensures all required fields are present and properly formatted.
    
    Args:
        parsed: Raw parsed data from LLM
        original_question: Original question (for filling gaps)
    
    Returns:
        Normalized structure
    """
    # Ensure required fields
    parsed.setdefault("raw_question", original_question)
    parsed.setdefault("problem_type", "equation")
    parsed.setdefault("goal", "solve")
    parsed.setdefault("unknowns", ["x"])
    
    # Normalize equations to consistent format
    equations = parsed.get("equations", [])
    normalized_equations = []
    
    for eq in equations:
        if isinstance(eq, dict) and "lhs" in eq and "rhs" in eq:
            normalized_equations.append({
                "lhs": str(eq["lhs"]),
                "rhs": str(eq["rhs"])
            })
        elif isinstance(eq, (list, tuple)) and len(eq) >= 2:
            normalized_equations.append({
                "lhs": str(eq[0]),
                "rhs": str(eq[1])
            })
        else:
            logger.warning(f"Skipping malformed equation: {eq}")
    
    parsed["equations"] = normalized_equations
    
    # Ensure unknowns exist
    if not parsed.get("unknowns") and normalized_equations:
        eq_tuples = [(eq["lhs"], eq["rhs"]) for eq in normalized_equations]
        parsed["unknowns"] = utils.detect_unknowns(eq_tuples)
    
    return parsed

def generate_python_from_json(structured: Dict[str, Any], *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate executable Python code from structured problem data.
    
    Pipeline:
    1. Try LLM code generation (if available)
    2. Validate generated code for safety
    3. Fallback to sympy_stub mode if needed
    
    Args:
        structured: Structured problem representation
        model_cfg: Optional model configuration
    
    Returns:
        Dictionary with:
        - mode: "llm_codegen" or "sympy_stub"
        - code: Generated Python code string (or None)
        - metadata: Generation details
    """
    logger.info("Generating solver code")
    if not isinstance(structured, dict):
        logger.error("Invalid structured input (not a dict)")
        return {
            "mode": "sympy_stub",
            "code": None,
            "metadata": {"error": "invalid_input_type"}
        }
    
    # Load code generation prompt
    prompt_template = _read_prompt("codegen.md")
    if not prompt_template:
        prompt_template = """Generate a Python script to solve this mathematical problem.

Requirements:
- Import ONLY: json, math, sympy
- No external data or network access
- Print result as JSON to stdout
- Include error handling
- Use standard output format:
  {
    "problem_type": "...",
    "solution": {"unknown": "x", "values": [...]},
    "answer": "..."
  }

Return ONLY raw Python code - no markdown fences, no explanations."""

    structured_json = json.dumps(structured, ensure_ascii=False, indent=2)
    full_prompt = f"{prompt_template}\n\n# PROBLEM:\n{structured_json}\n\n# CODE:\n"

    # Try LLM code generation
    if PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini" and client:
        try:
            logger.debug("Attempting LLM code generation...")
            response_text = _call_gemini(
                full_prompt,
                model=CODEGEN_MODEL,
                temperature=CODEGEN_TEMP
            )
            
            # Clean up markdown fences
            code = unwrap_markdown_code(response_text)

            # Basic safety pre-check
            is_safe, issues = check_code_safety_basic(code)
            
            if not is_safe:
                logger.warning(f"Generated code failed basic safety check: {issues}")
                return {
                    "mode": "sympy_stub",
                    "code": None,
                    "metadata": {
                        "reason": "safety_precheck_failed",
                        "issues": issues
                    }
                }
            
            if not code.strip():
                logger.warning("LLM returned empty code")
                return {
                    "mode": "sympy_stub",
                    "code": None,
                    "metadata": {"reason": "empty_code"}
                }
        
            logger.info(f"✓ Code generation successful ({len(code)} chars, {code.count(chr(10))+1} lines)")
            return {
                "mode": "llm_codegen",
                "code": code,
                "metadata": {
                    "model": CODEGEN_MODEL,
                    "code_length": len(code),
                    "code_lines": code.count("\n") + 1
                }
            }
        
        except Exception as e:
            logger.error(f"Code generation failed: {e}")

    # Fallback
    logger.info("Using sympy_stub fallback")
    return {
        "mode": "sympy_stub",
        "code": None,
        "metadata": {"reason": "llm_unavailable_or_failed"}
    }


def explain_result(
    question: str,
    execution_output: Dict[str, Any],
    generated_code: Optional[str] = None,
    *,
    model_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate human-readable HTML explanation of solution.
    
    Pipeline:
    1. Try LLM explanation (if available)
    2. Fallback to deterministic HTML generation
    
    Args:
        question: Original question
        execution_output: Execution/solution result
        generated_code: Optional solver code for context
        model_cfg: Optional model configuration
    
    Returns:
        Dictionary with:
        - steps_html: HTML explanation
        - confidence: Confidence score (0.0-1.0)
    """
    logger.info("Generating solution explanation")

    if not execution_output:
        return {
            "steps_html": "<p>No result to explain.</p>",
            "confidence": 0.0
        }
    # Load explanation prompt
    prompt_template = _read_prompt("explain.md")
    if not prompt_template:
        prompt_template = """Generate a clear, step-by-step explanation of this solution.

Requirements:
- Target audience: High school student
- Format: Clean HTML with proper structure
- Include: Problem statement, solution steps, final answer
- Be concise but thorough

Return ONLY the HTML explanation, no markdown fences."""

    # Prepare context
    result_json = json.dumps(execution_output, ensure_ascii=False, indent=2)
    code_context = ""
    
    if generated_code:
        code_clean = utils.unwrap_markdown_code(generated_code)
        # Limit code length to avoid token limits
        code_context = f"\n\n# SOLVER CODE:\n```python\n{code_clean[:2500]}\n```"
    
    full_prompt = f"""{prompt_template}

# QUESTION:
{question}

# SOLUTION RESULT:
{result_json}{code_context}

# HTML EXPLANATION:
"""
     # Try LLM explanation
    if PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini" and client:
        try:
            logger.debug("Attempting LLM explanation generation...")
            response_text = _call_gemini(
                full_prompt,
                model=EXPLAIN_MODEL,
                temperature=EXPLAIN_TEMP
            )
            
            if response_text and response_text.strip():
                logger.info(f"✓ Explanation generated ({len(response_text)} chars)")
                return {
                    "steps_html": response_text.strip(),
                    "confidence": 1.0
                }
        
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")

    # Fallback to deterministic explanation
    logger.info("Using fallback explanation generator")
    return generate_fallback_explanation(question, execution_output, generated_code)

def generate_fallback_explanation(
    question: str,
    result: Dict[str, Any],
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate basic HTML explanation when LLM is unavailable.
    
    Args:
        question: Original question
        result: Solution result
        code: Optional solver code
    
    Returns:
        Dictionary with steps_html and confidence
    """
    parts = []
    
    # Container start
    parts.append("<div style='font-family: Arial, sans-serif; line-height: 1.8; color: #333;'>")
    
    # Problem statement
    parts.append("<h3 style='color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 0.5rem;'>Problem</h3>")
    parts.append(f"<p><strong>{question}</strong></p>")
    
    # Equations (if available)
    structured = result.get("structured", {})
    equations = structured.get("equations", []) if isinstance(structured, dict) else []
    
    if equations:
        parts.append("<h3 style='color: #1976d2;'>Equation(s)</h3>")
        parts.append("<ol style='padding-left: 2rem;'>")
        for eq in equations:
            if isinstance(eq, dict):
                lhs = eq.get("lhs", "")
                rhs = eq.get("rhs", "")
                if lhs and rhs:
                    parts.append(f"<li style='margin: 0.5rem 0;'><code>{lhs} = {rhs}</code></li>")
        parts.append("</ol>")
    
    # Solution
    solution = result.get("solution")
    if solution and isinstance(solution, dict):
        unknown = solution.get("unknown", "x")
        values = solution.get("values", [])
        
        if values:
            parts.append("<h3 style='color: #1976d2;'>Solution</h3>")
            if len(values) == 1:
                parts.append(
                    f"<p style='font-size: 1.2rem; font-weight: bold; color: #2e7d32;'>"
                    f"{unknown} = {values[0]}</p>"
                )
            else:
                values_str = ", ".join(str(v) for v in values)
                parts.append(
                    f"<p style='font-size: 1.2rem; font-weight: bold; color: #2e7d32;'>"
                    f"{unknown} = {values_str}</p>"
                )
    
    # Validation status
    validation = result.get("validation", {})
    if validation.get("ok"):
        parts.append(
            "<p style='color: #2e7d32;'>"
            "✓ Solution verified by substitution into original equation(s)</p>"
        )
    
    # Note
    parts.append(
        "<p style='color: #666; font-size: 0.9rem; font-style: italic; "
        "margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd;'>"
        "Note: This explanation was generated using deterministic methods.</p>"
    )
    
    parts.append("</div>")
    
    return {
        "steps_html": "".join(parts),
        "confidence": 0.5
    }

# ============================================================================
# CLI FOR TESTING
# ============================================================================
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="LLM Wrapper CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute") 

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse question to JSON")
    parse_parser.add_argument("question", nargs="+", help="Question text")
    
    # Codegen command
    codegen_parser = subparsers.add_parser("codegen", help="Generate code from JSON")
    codegen_parser.add_argument("jsonfile", help="Path to structured JSON file")
    codegen_parser.add_argument("--output", "-o", help="Output file for generated code")
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Generate explanation")
    explain_parser.add_argument("question", nargs="+", help="Question text")
    explain_parser.add_argument("resultfile", help="Path to result JSON file")
    explain_parser.add_argument("--code", "-c", help="Optional path to solver code")
    explain_parser.add_argument("--output", "-o", help="Output file for HTML")
    
    args = parser.parse_args()

    if args.command == "parse":
        question = " ".join(args.question)
        result = parse_text_to_json(question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.command == "codegen":
        with open(args.jsonfile, "r", encoding="utf-8") as f:
            structured = json.load(f)
        result = generate_python_from_json(structured)
        
        if args.output and result.get("code"):
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result["code"])
            print(f"Code written to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "explain":
        question = " ".join(args.question)
        
        with open(args.resultfile, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        code = None
        if args.code:
            try:
                with open(args.code, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                logger.warning(f"Could not read code file: {e}")
        
        result = explain_result(question, result_data, code)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result["steps_html"])
            print(f"Explanation written to: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        parser.print_help()
