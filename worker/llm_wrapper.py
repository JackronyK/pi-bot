# worker/llm_wrapper.py
"""
LLM wrapper using Google Gemini (genai). Supports separate models for:
 - parser
 - codegen
 - explainer

Environment variables (defaults shown):
PIBOT_GEMINI_PARSER_MODEL=gemini-2.5-flash
PIBOT_GEMINI_CODEGEN_MODEL=gemini-2.5-pro
PIBOT_GEMINI_EXPLAIN_MODEL=gemini-2.5-pro

PIBOT_GEMINI_PARSER_TEMP=0.0
PIBOT_GEMINI_CODEGEN_TEMP=0.0
PIBOT_GEMINI_EXPLAIN_TEMP=0.2

GEMINI_API_KEY must be set in environment for genai.Client() to work.
"""
from __future__ import annotations

import os
import json
import re
import logging
from typing import Any, Dict, Optional

# local helpers (heuristics / sympy stub)
import utils

# genai client
from google import genai

logger = logging.getLogger("worker.llm_wrapper")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# --------------------
# Configuration
# --------------------
PIBOT_USE_LLM = os.environ.get("PIBOT_USE_LLM", "false").lower() in ("1", "true", "yes")
PIBOT_LLM_PROVIDER = os.environ.get("PIBOT_LLM_PROVIDER", "gemini").lower()  # 'gemini' or 'mock'
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Gemini / provider settings (optional)
PARSER_MODEL = os.environ.get("PIBOT_GEMINI_PARSER_MODEL", "gemini-2.5-flash")
CODEGEN_MODEL = os.environ.get("PIBOT_GEMINI_CODEGEN_MODEL", "gemini-2.5-pro")
EXPLAIN_MODEL = os.environ.get("PIBOT_GEMINI_EXPLAIN_MODEL", "gemini-2.5-pro")

# LLM runtime params
PARSER_TEMP = float(os.environ.get("PIBOT_GEMINI_PARSER_TEMP", "0.0"))
CODEGEN_TEMP = float(os.environ.get("PIBOT_GEMINI_CODEGEN_TEMP", "0.0"))
EXPLAIN_TEMP = float(os.environ.get("PIBOT_GEMINI_EXPLAIN_TEMP", "0.2"))

# quick forbidden imports for fast pre-check (additional static scan should run later)
_FORBIDDEN_IMPORT_RE = re.compile(r"\b(import|from)\s+(os|sys|subprocess|socket|shutil|ctypes|multiprocessing|threading|requests)\b", flags=re.IGNORECASE)

# -------------------------
# Client init
# -------------------------
# genai client will pick API key from env variable (GEMINI_API_KEY) as library expects
try:
    client = genai.Client()
except Exception:
    client = None
    logger.debug("genai.Client() init failed or not available in this environment - LLM calls will error if attempted")

# --------------------
# Helpers
# --------------------
def _read_prompt(name: str) -> str:
    path = os.path.join(PROMPT_DIR, name)
    try:
        with open(path, "r", encoding="utf8") as f:
            return f.read()
    except FileNotFoundError:
        logger.debug("Prompt file not found: %s", path)
        return ""
    except Exception as e:
        logger.exception("Error reading prompt file %s: %s", path, e)
        return ""

def _call_gemini(prompt: str, model: str, temperature: float) -> str:
    """
    Call Gemini via genai client and return text output (best-effort).
    Attempts to pass temperature; if genai wrapper doesn't support the kw, falls back.
    """
    if client is None:
        raise RuntimeError("genai client not initialized; set GEMINI_API_KEY and install google-genai library")

    logger.debug("Calling gemini model=%s temp=%s len_prompt=%d", model, temperature, len(prompt or ""))
    try:
        # try calling with temperature; genai wrappers differ, so handle both styles
        try:
            response = client.models.generate_content(model=model, contents=prompt,) #temperature=temperature)
        except TypeError:
            # some versions accept generation_config
            response = client.models.generate_content(model=model, contents=prompt)  # fallback
        # prefer .text if available
        text = getattr(response, "text", None)
        if text:
            return text
        # some wrappers put output in response.generations / response.output etc.
        # attempt some common access patterns:
        if hasattr(response, "generations"):
            try:
                gens = response.generations
                if isinstance(gens, (list, tuple)) and len(gens) > 0:
                    # try to get text from first generation
                    g0 = gens[0]
                    if isinstance(g0, dict) and "text" in g0:
                        return g0["text"]
                    elif hasattr(g0, "text"):
                        return g0.text
            except Exception:
                pass
        # fallback to stringified response
        resp_str = str(response)
        logger.debug("genai response (fallback str): %s", resp_str[:1000])
        return resp_str
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        raise

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to find a JSON object inside model text. Returns dict or None.
    This is tolerant: strips surrounding backticks, finds first {...} block.
    """
    if not text:
        return None
    # remove code fences if present
    t = re.sub(r"```(?:json|js)?\n", "", text, flags=re.IGNORECASE)
    t = t.strip(" `\n")
    # direct load attempt
    try:
        return json.loads(t)
    except Exception:
        # try to find the first {...} substring (balanced brace heuristic)
        m = re.search(r"\{", t)
        if not m:
            return None
        start = m.start()
        # naive balancing to find matching closing brace
        depth = 0
        end = None
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if not end:
            return None
        candidate = t[start:end]
        try:
            return json.loads(candidate)
        except Exception:
            # last resort: try to progressively trim trailing chars
            for eidx in range(len(candidate), start, -1):
                try:
                    return json.loads(candidate[:eidx])
                except Exception:
                    continue
    return None

def unwrap_markdown_code(s: str) -> str:
    """
    Remove markdown fences and any leading/trailing explanatory lines.
    Returns cleaned code string.
    """
    if not s:
        return ""
    s = s.strip()
    # fenced triple backticks with optional language spec
    m = re.search(r"```(?:python|py)?\n(.+?)\n```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # inline fenced block
    one = re.search(r"```(.+?)```", s, flags=re.DOTALL)
    if one:
        return one.group(1).strip()
    # if the model returned code surrounded by ``` on separate lines at ends
    s = re.sub(r"^```(?:python|py)?\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```$", "", s, flags=re.IGNORECASE)
    return s.strip()

# -------------------------
# Public API: parser / codegen / explain
# -------------------------
def parse_text_to_json(question: str, *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse natural-language `question` to structured JSON using Gemini parser model.
    If the model output is invalid JSON, fall back to heuristic minimal structure.
    """
    logger.info("parse_text_to_json called; parser_model=%s PIBOT_USE_LLM=%s", PARSER_MODEL, PIBOT_USE_LLM)
    prompt_template = _read_prompt("parser.md") or (
        "Parse the following math question into JSON with keys: problem_type, raw_question, goal, equations (list of {lhs,rhs}), unknowns (list)."
    )
    prompt = f"{prompt_template}\n\n# QUESTION:\n{question}\n\n# OUTPUT JSON:\n"

    if PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini":
        try:
            raw = _call_gemini(prompt, model=PARSER_MODEL, temperature=PARSER_TEMP)
            parsed = _extract_json_from_text(raw)
            if parsed is None:
                logger.warning("Parser model did not return valid JSON — falling back to heuristics. Raw output snippet:\n%s", (raw or "")[:1000])
            else:
                # Normalize equations: allow either list-of-pairs or list-of-objects
                eqs = parsed.get("equations", [])
                normalized = []
                for e in eqs:
                    if isinstance(e, dict) and "lhs" in e and "rhs" in e:
                        normalized.append({"lhs": e["lhs"], "rhs": e["rhs"]})
                    elif isinstance(e, (list, tuple)) and len(e) >= 2:
                        normalized.append({"lhs": e[0], "rhs": e[1]})
                parsed["equations"] = normalized
                parsed.setdefault("raw_question", question)
                parsed.setdefault("goal", parsed.get("goal", "solve"))
                # ensure unknowns exist
                if not parsed.get("unknowns"):
                    parsed["unknowns"] = utils.detect_unknowns([(d["lhs"], d["rhs"]) for d in normalized]) if normalized else ["x"]
                return parsed
        except Exception as e:
            logger.exception("Parser LLM error: %s", e)

    # Heuristic fallback
    try:
        eqs = utils.extract_equations(question)
        unknowns = utils.detect_unknowns(eqs) if eqs else ["x"]
        # convert to list-of-dicts for consistency
        eqs_norm = [{"lhs": l, "rhs": r} for l, r in eqs]
        return {"raw_question": question, "equations": eqs_norm, "unknowns": unknowns, "goal": "solve", "notes": "heuristic-fallback"}
    except Exception as e:
        logger.exception("Heuristic fallback failed: %s", e)
        return {"raw_question": question, "equations": [], "unknowns": ["x"], "goal": "solve", "notes": "error-fallback"}

def generate_python_from_json(structured: Dict[str, Any], *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate Python code from structured JSON using Gemini codegen model.
    Returns: {"mode": "llm_codegen"|"sympy_stub", "code": <str>|None, "metadata": {...}}
    """
    logger.info("generate_python_from_json called; codegen_model=%s PIBOT_USE_LLM=%s", CODEGEN_MODEL, PIBOT_USE_LLM)
    if not isinstance(structured, dict):
        return {"mode": "sympy_stub", "code": None, "metadata": {"error": "invalid_structured"}}

    prompt_template = _read_prompt("codegen.md") or (
        "Generate a Python script that reads no external data, solves the following structured problem, and prints a JSON result to stdout.\n"
        "Return only raw Python code (no markdown fences, no explanations). Allowed imports: json, math, sympy."
    )
    structured_str = json.dumps(structured, ensure_ascii=False, indent=2)
    prompt = f"{prompt_template}\n\n# STRUCTURED_INPUT:\n{structured_str}\n\n# PYTHON SCRIPT (only code, no commentary):\n"

    if PIBOT_USE_LLM and PIBOT_LLM_PROVIDER == "gemini":
        try:
            raw = _call_gemini(prompt, model=CODEGEN_MODEL, temperature=CODEGEN_TEMP)
            # unwrap fencing if present
            cleaned = unwrap_markdown_code(raw)
            # Quick static heuristic: disallow obvious forbidden imports
            if _FORBIDDEN_IMPORT_RE.search(cleaned):
                logger.warning("Codegen returned unsafe imports - rejecting generated code.")
                return {"mode": "sympy_stub", "code": None, "metadata": {"reason": "unsafe_import_detected"}}
            if not cleaned.strip():
                logger.warning("Codegen returned empty code - falling back.")
                return {"mode": "sympy_stub", "code": None, "metadata": {"reason": "empty_codegen"}}
            return {"mode": "llm_codegen", "code": cleaned, "metadata": {"model": CODEGEN_MODEL}}
        except Exception as e:
            logger.exception("Codegen LLM failed: %s", e)
            return {"mode": "sympy_stub", "code": None, "metadata": {"error": str(e)}}

    # Default safe fallback to sympy
    return {"mode": "sympy_stub", "code": None, "metadata": {"reason": "llm_disabled_or_provider_not_gemini"}}

def explain_result(question: str,
                   execution_output: Dict[str, Any],
                   generated_code: Optional[str] = None,
                   *,
                   model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produce an HTML explanation using Gemini explainer model.

    Accepts optional `generated_code` (string) so the explainer can use the actual solver
    script as context when producing step-by-step instructions.

    Fallback behavior:
      - If the model returns empty/whitespace, build a simple deterministic HTML explanation
        from execution_output (equations + solution).
    """
    logger.info("explain_result called; explain_model=%s", EXPLAIN_MODEL)
    if not execution_output:
        return {"steps_html": "<p>No result to explain.</p>", "confidence": 0.0}

    # Compose the prompt: include question, result JSON and optionally the generated code
    prompt_template = _read_prompt("explain.md") or (
        "Given the QUESTION, the SOLVER_RESULT (JSON) and the SOLVER_CODE (if present), "
        "produce a clear step-by-step explanation in HTML for a high-school student. "
        "Return only the explanation text (HTML)."
    )

    result_str = json.dumps(execution_output, ensure_ascii=False, indent=2)
    code_str = generated_code or ""
    # unwrap code fences if user passed the file contents
    code_str = unwrap_markdown_code(code_str)

    prompt_parts = [
        prompt_template,
        "\n# QUESTION:\n",
        question,
        "\n\n# RESULT_JSON:\n",
        result_str,
    ]
    if code_str:
        prompt_parts += ["\n\n# SOLVER_CODE (for context):\n", code_str[:4000]]  # limit length to be safe
    prompt_parts += ["\n\n# EXPLANATION (HTML):\n"]

    prompt = "".join(prompt_parts)

    try:
        raw = _call_gemini(prompt, model=EXPLAIN_MODEL, temperature=EXPLAIN_TEMP)
        # If model returns nothing meaningful, fallback to deterministic builder
        if raw is None:
            raw = ""
        raw = raw.strip()
        if raw == "":
            logger.warning("Explainer model returned empty output; using fallback explanation.")
            raise RuntimeError("empty_explainer_output")
        # Some models might return fenced Markdown with JSON or code; we assume it's HTML/text
        steps_html = raw
        return {"steps_html": steps_html, "confidence": 1.0}
    except Exception as e:
        logger.exception("Explainer LLM failed or returned empty: %s", e)

        # Fallback: build a deterministic HTML explanation from execution_output + (optional) code
        try:
            # Prefer structured.solutions if present
            sol = None
            if isinstance(execution_output.get("structured"), dict):
                sol = execution_output["structured"].get("solutions")
            if sol is None:
                sol = execution_output.get("answer") or execution_output.get("solutions")

            # Build simple explanation html
            parts = []
            parts.append(f"<p><strong>Question:</strong> {question}</p>")
            # If structured contains equations, show them
            structured = execution_output.get("structured", {})
            eqs = structured.get("equations") if isinstance(structured, dict) else None
            if not eqs:
                # try to infer simple equation(s) from execution_output.answer or steps
                eqs = None

            if eqs:
                parts.append("<p><strong>Equation(s):</strong></p><ol>")
                # eqs may be list of lists or dicts; attempt to render
                for e in eqs:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        lhs, rhs = e
                        parts.append(f"<li>$$ {lhs} = {rhs} $$</li>")
                    elif isinstance(e, dict):
                        lhs = e.get("lhs"); rhs = e.get("rhs")
                        if lhs and rhs:
                            parts.append(f"<li>$$ {lhs} = {rhs} $$</li>")
                        else:
                            parts.append(f"<li>{json.dumps(e)}</li>")
                    else:
                        parts.append(f"<li>{e}</li>")
                parts.append("</ol>")

            # solution
            parts.append("<p><strong>Solution:</strong></p>")
            if isinstance(sol, (list, tuple)):
                parts.append("<p>" + ", ".join(str(s) for s in sol) + "</p>")
            else:
                parts.append(f"<p>{sol}</p>")

            # include a small deterministic step-by-step if possible
            # e.g., if a linear equation of form ax + b = c, show isolate x
            # Simple heuristic: if single equation string present, try to perform naive steps
            # (we won't run external code here)
            try:
                # add code snippet context if present
                if code_str:
                    parts.append("<p><strong>Solver code (summary):</strong></p>")
                    # avoid dumping giant code; show first 20 lines
                    snippet = "\n".join(code_str.splitlines()[:20])
                    parts.append(f"<pre>{snippet}</pre>")
            except Exception:
                pass

            parts.append("<p>Note: explanation generated deterministically as fallback.</p>")

            steps_html = "".join(parts)
            return {"steps_html": steps_html, "confidence": 0.0}
        except Exception as e2:
            logger.exception("Fallback explainer building failed: %s", e2)
            return {"steps_html": "<p>Could not generate explanation.</p>", "confidence": 0.0}



# -------------------------
# CLI for quick local testing
# -------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("parse")
    p.add_argument("question", nargs="+")
    c = sub.add_parser("codegen")
    c.add_argument("jsonfile")
    e = sub.add_parser("explain")
    e.add_argument("question", nargs="+")
    e.add_argument("resultfile")
    e.add_argument("--code", "-c", help="Optional path to generated solver code (python)")


    args = parser.parse_args()
    if args.cmd == "parse":
        q = " ".join(args.question)
        out = parse_text_to_json(q)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "codegen":
        with open(args.jsonfile, "r", encoding="utf8") as f:
            structured = json.load(f)
        out = generate_python_from_json(structured)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "explain":
        q = " ".join(args.question)
        # load result JSON
        with open(args.resultfile, "r", encoding="utf8") as f:
            res = json.load(f)
        generated_code = None
        if getattr(args, "code", None):
            try:
                with open(args.code, "r", encoding="utf8") as cf:
                    generated_code = cf.read()
            except Exception as e:
                logger.warning("Could not read code file %s: %s", args.code, e)
        out = explain_result(q, res, generated_code)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
