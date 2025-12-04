# worker/llm_wrapper.py
"""
Lightweight LLM wrapper (Sprint 1.1).

Public API:
  - parse_text_to_json(question) -> structured dict
  - generate_python_from_json(structured) -> {"mode":..., "code":..., "metadata": {...}}
  - explain_result(question, execution_output) -> {"steps_html": ..., "confidence": ...}

Design goals:
- Safe defaults: when actual LLM/model is not enabled, return deterministic, safe fallbacks (sympy_stub / llm_stub).
- Small, clear extension points for plugging in a real local model (gpt4all/StarCoder/transformers).
- No heavy dependencies in this file; uses worker.utils for heuristics.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional

import utils

logger = logging.getLogger("worker.llm_wrapper")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# Config via environment variables
PIBOT_USE_LLM = os.environ.get("PIBOT_USE_LLM", "true").lower() in ("1", "true", "yes")
PIBOT_MODEL = os.environ.get("PIBOT_MODEL", "local_stub")  # placeholder name
PIBOT_MODEL_PATH = os.environ.get("PIBOT_MODEL_PATH", "")
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


# -------------------------
# Internal helpers
# -------------------------
def _read_prompt(name: str) -> str:
    """Read a prompt file from worker/prompts/<name> if present, else return empty string."""
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


def _model_invoke_stub(prompt: str, *, role: str = "gen") -> str:
    """
    Placeholder model call. For sprint 1.1 we intentionally do NOT call any external model.
    When you wire a real local model later, replace this function with the actual invocation.
    """
    logger.debug("LLM stub invoked (role=%s). Prompt length=%d", role, len(prompt or ""))
    # deterministic stub response (useful for tests)
    if role == "parse":
        # return a minimal JSON-like text (we won't parse here; parse_text_to_json uses utils)
        return "{}"
    if role == "code":
        return "# generated-code-stub\nprint('hello from llm stub')\n"
    if role == "explain":
        return "<p>LLM explanation (stub)</p>"
    return ""


# -------------------------
# Public API
# -------------------------
def parse_text_to_json(question: str, *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse a natural-language question into a structured JSON object.
    Sprint-1 behaviour: use heuristic fallback (worker.utils.extract_equations).
    Later: call model parser with a prompt, then parse model JSON safely.
    """
    logger.info("parse_text_to_json called; PIBOT_USE_LLM=%s", PIBOT_USE_LLM)
    # Use deterministic heuristic first (robust for Sprint 1)
    try:
        eqs = utils.extract_equations(question)
        unknowns = utils.detect_unknowns(eqs) if eqs else ["x"]
        structured = {
            "raw_question": question,
            "equations": eqs,
            "unknowns": unknowns,
            "goal": "solve",
            "notes": "heuristic-fallback",
        }

        # if LLM is enabled you could attempt model parsing here, but keep fallback stable
        if PIBOT_USE_LLM:
            # Example: load parsing prompt and call model (stubbed now)
            parse_prompt = _read_prompt("parser.md") or f"Parse this math question into JSON: {question}"
            try:
                model_out = _model_invoke_stub(parse_prompt, role="parse")
                # If model returns JSON, attempt to parse it safely
                # NOTE: only attempt if non-empty and appears JSON-ish
                if model_out and model_out.strip().startswith("{"):
                    parsed = json.loads(model_out)
                    # Merge/validate parsed (best-effort)
                    parsed.setdefault("raw_question", question)
                    parsed.setdefault("goal", "solve")
                    return parsed
            except Exception:
                logger.debug("LLM parser stub failed; using heuristic fallback")

        return structured

    except Exception as e:
        logger.exception("Heuristic parser failed: %s", e)
        return {"raw_question": question, "equations": [], "unknowns": ["x"], "goal": "solve", "notes": "error-fallback"}


def generate_python_from_json(structured: Dict[str, Any], *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate Python solver code from structured JSON.
    Return dict: {"mode": <label>, "code": <text or None>, "metadata": {...}}

    Sprint-1 behaviour:
      - If LLM disabled -> return sympy_stub
      - If LLM enabled but no model impl -> return sympy_stub (safe)
    Later:
      - Call model to synthesize code, run static safety checks, return "llm_codegen" when ready.
    """
    logger.info("generate_python_from_json called; PIBOT_USE_LLM=%s", PIBOT_USE_LLM)
    # Quick validation of structured
    if not isinstance(structured, dict):
        return {"mode": "sympy_stub", "code": None, "metadata": {"error": "invalid_structured"}}

    # If LLM usage is explicitly requested via structured or model_cfg, you'd call model
    if PIBOT_USE_LLM:
        # load codegen template/prompt if present
        code_prompt = _read_prompt("codegen.md") or f"# generate python for {structured.get('raw_question','')}"
        try:
            code_text = _model_invoke_stub(code_prompt, role="code")
            # TODO: run safety scan here before returning "llm_codegen" (or orchestrator will check)
            return {"mode": "llm_codegen", "code": code_text, "metadata": {"model": PIBOT_MODEL}}
        except Exception as e:
            logger.exception("LLM codegen stub failed: %s", e)
            return {"mode": "sympy_stub", "code": None, "metadata": {"error": str(e)}}

    # Default safe fallback
    return {"mode": "sympy_stub", "code": None, "metadata": {"reason": "sprint1-fallback"}}


def explain_result(question: str, execution_output: Dict[str, Any], *, model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produce a student-friendly explanation (HTML steps) given the execution output.
    Sprint-1 behaviour: prefer existing `steps_html` from execution_output; else generate a small wrapper.
    Later: call LLM explainer to produce richer step-by-step breakdown.
    """
    logger.info("explain_result called; PIBOT_USE_LLM=%s", PIBOT_USE_LLM)
    if not execution_output:
        return {"steps_html": "<p>No result to explain.</p>", "confidence": 0.0}

    # If there's already steps_html produced by sympy stub, keep it
    steps = execution_output.get("steps_html") or execution_output.get("steps") or execution_output.get("answer")
    if isinstance(steps, list):
        steps_html = "".join(f"<p>{s}</p>" for s in steps)
    elif isinstance(steps, str):
        steps_html = steps
    else:
        steps_html = f"<p>Answer: {execution_output.get('answer')}</p>"

    # If LLM enabled we could enrich the steps (stubbed here)
    if PIBOT_USE_LLM:
        explain_prompt = _read_prompt("explain.md") or f"Explain this result for a student: {execution_output.get('answer')}"
        try:
            explain_text = _model_invoke_stub(explain_prompt, role="explain")
            if explain_text:
                # model returns HTML/text; prefer it
                steps_html = explain_text
        except Exception:
            logger.debug("LLM explain stub failed; returning existing steps")

    return {"steps_html": steps_html, "confidence": 1.0}


# -------------------------
# Small run-time helpers (stubs) used for local testing
# -------------------------
def run_llm_stub(code: str, question: str) -> Dict[str, Any]:
    """
    Simulate executing LLM-generated solver code (used in tests).
    This intentionally does NOT execute arbitrary code. Instead it returns a safe stub result.
    """
    logger.debug("run_llm_stub invoked for question: %s", question)
    return {
        "answer": "[LLM_stub_result]",
        "structured": {"type": "llm_stub", "solutions": "[LLM_stub_result]"},
        "steps_html": f"<p>LLM stub processed question: {question}</p>",
        "solved_by": "llm_stub",
        "validation": {"ok": True, "method": "stub_validation"},
    }


def generate_stub_code(question: str) -> Dict[str, str]:
    """
    Generate a trivially safe Python snippet for testing (llm_codegen_stub).
    The code does not do dangerous operations — only prints a placeholder.
    """
    code_template = (
        "# Safe stub code generated for question\n"
        "def solve():\n"
        f"    # placeholder solution for: {question!s}\n"
        "    print(42)\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n"
    )
    return {"mode": "llm_codegen_stub", "code": code_template}


# -------------------------
# CLI for manual dev/testing
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM wrapper test CLI (Sprint 1.1)")
    sub = parser.add_subparsers(dest="cmd")

    p_parse = sub.add_parser("parse", help="Run parse_text_to_json")
    p_parse.add_argument("question", help="Question text to parse")

    p_codegen = sub.add_parser("codegen", help="Run generate_python_from_json")
    p_codegen.add_argument("jsonfile", help="Path to structured-json file")

    p_explain = sub.add_parser("explain", help="Run explain_result")
    p_explain.add_argument("question", help="Original question")
    p_explain.add_argument("resultfile", help="Path to executor JSON result file")

    args = parser.parse_args()

    if args.cmd == "parse":
        out = parse_text_to_json(args.question)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "codegen":
        with open(args.jsonfile, "r", encoding="utf8") as f:
            structured = json.load(f)
        out = generate_python_from_json(structured)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "explain":
        with open(args.resultfile, "r", encoding="utf8") as f:
            result = json.load(f)
        out = explain_result(args.question, result)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
