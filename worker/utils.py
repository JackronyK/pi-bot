# worker/utils.py
"""
Utility helpers for PiBot worker:
- Redis job record helpers
- Text parsing / normalization
- SymPy equation builders and solvers
- Validation and step formatting
"""

from __future__ import annotations

import re
import json
import time
from typing import List, Any, Dict, Tuple, Optional

import redis
from sympy import symbols, Eq, latex, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy.core.sympify import SympifyError

# Redis connection helper (same config used elsewhere)
REDIS_URL = "redis://redis:6379"
redis_conn = redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------
# Job storage helpers
# ---------------------------
def get_job_record(job_id: str) -> Optional[Dict[str, Any]]:
    """Return parsed job record from redis or None if missing."""
    key = f"job:{job_id}"
    raw = redis_conn.get(key)
    return json.loads(raw) if raw else None


def update_job_record(job_id: str, patch: dict) -> None:
    """Patch an existing job record in redis (shallow update)."""
    key = f"job:{job_id}"
    raw = redis_conn.get(key)
    if raw is None:
        return
    job = json.loads(raw)
    job.update(patch)
    redis_conn.set(key, json.dumps(job))


# ---------------------------
# Text normalization & parsing helpers
# ---------------------------
def _strip_leading_instruction(text: str) -> str:
    """
    Remove common natural-language prefixes like:
      - "Solve for x: ...", "Find x: ...", "Compute: ...", "Evaluate: ..."
    Returns the rest (if pattern matched) or original text.
    """
    t = text.strip()
    m = re.match(
        r"(?is)^\s*(solve(?: for [a-zA-Z]\w*)?|find(?: for [a-zA-Z]\w*)?|compute|evaluate|simplify)\b[:\s]*(.*)$",
        t,
    )
    if m:
        remainder = m.group(2).strip()
        if remainder:
            return remainder
    return t


def preprocess_equation_text(text: str) -> str:
    """Basic sanitation: replace caret with power operator and remove thousands separators."""
    t = text.strip()
    t = t.replace("^", "**")
    t = t.replace(",", "")
    return t


def extract_equations(text: str) -> List[Tuple[str, str]]:
    """
    Extract one or more equations from text.
    Returns list of (lhs_str, rhs_str) tuples or [] if none found.
    Splits on semicolon/newline/| to support systems.
    """
    t = _strip_leading_instruction(text)
    # split common list separators
    eq_parts = re.split(r"(?:;|\n|\|)", t)
    eqs: List[Tuple[str, str]] = []
    for part in eq_parts:
        if "=" in part:
            lhs, rhs = part.split("=", 1)
            eqs.append((lhs.strip(), rhs.strip()))

    # fallback: if nothing found but '=' exists in whole text
    if not eqs and "=" in t:
        lhs, rhs = t.split("=", 1)
        eqs.append((lhs.strip(), rhs.strip()))
    return eqs


# ---------------------------
# Symbol detection
# ---------------------------
def detect_unknowns(equations: List[Tuple[str, str]]) -> List[str]:
    """
    Heuristic: look for single-letter tokens (x, y, z). If none found, default to ['x'].
    Returns a list of variable names (strings).
    """
    vars_set = set()
    for lhs, rhs in equations:
        tokens = re.findall(r"[a-zA-Z_]\w*", lhs + " " + rhs)
        for tok in tokens:
            if len(tok) == 1 and tok.isalpha():
                vars_set.add(tok)
    if not vars_set:
        vars_set = {"x"}
    return list(vars_set)


# ---------------------------
# SymPy builders / parsing
# ---------------------------
def build_sympy_eqs(equations: List[Tuple[str, str]]) -> List[Eq]:
    """
    Build sympy Eq objects from list of (lhs_str, rhs_str).
    Uses SymPy's implicit_multiplication_application so inputs like '2x' parse.
    Raises ValueError with a helpful message if parsing fails.
    """
    eq_objs: List[Eq] = []
    transformations = standard_transformations + (implicit_multiplication_application,)

    for lhs_str, rhs_str in equations:
        lhs_text = preprocess_equation_text(lhs_str)
        rhs_text = preprocess_equation_text(rhs_str)
        try:
            lhs = parse_expr(lhs_text, transformations=transformations, evaluate=True)
        except (SympifyError, SyntaxError, ValueError) as e:
            raise ValueError(f"Could not parse left side '{lhs_str}' -> '{lhs_text}': {e}")
        try:
            rhs = parse_expr(rhs_text, transformations=transformations, evaluate=True)
        except (SympifyError, SyntaxError, ValueError) as e:
            raise ValueError(f"Could not parse right side '{rhs_str}' -> '{rhs_text}': {e}")
        eq_objs.append(Eq(lhs, rhs))

    return eq_objs


# ---------------------------
# Validation helpers
# ---------------------------
def validate_solution(eq_objs: List[Eq], solutions: Any, unknown_syms: List[Any]) -> bool:
    """
    For algebraic equations, check that each candidate solution satisfies the equations.
    - solutions can be dict, list/tuple/set, or single value.
    - unknown_syms: list of sympy Symbols (or their string names)
    Returns True if validation passes for at least one candidate mapping.
    """
    try:
        # normalize unknown_syms to strings (if symbols passed)
        unk_names = [str(s) for s in unknown_syms]

        # normalize solutions into list-of-mapping form
        sol_map_list: List[Dict[str, Any]] = []
        if isinstance(solutions, dict):
            sol_map_list = [solutions]
        elif isinstance(solutions, (list, tuple, set)):
            if len(unk_names) == 1 and not (solutions and isinstance(next(iter(solutions)), dict)):
                # list of simple values -> wrap each
                for val in solutions:
                    sol_map_list.append({unk_names[0]: val})
            else:
                # list of mappings maybe
                for s in solutions:
                    if isinstance(s, dict):
                        sol_map_list.append(s)
        else:
            # single scalar
            sol_map_list = [{unk_names[0]: solutions}]

        # Try each candidate mapping
        for sol_map in sol_map_list:
            all_zero = True
            for eq in eq_objs:
                lhs = eq.lhs.subs(sol_map)
                rhs = eq.rhs.subs(sol_map)
                diff = simplify(lhs - rhs)
                if diff == 0:
                    continue
                try:
                    if abs(float(diff.evalf())) < 1e-6:
                        continue
                except Exception:
                    all_zero = False
                    break
            if all_zero:
                return True
        return False
    except Exception:
        return False


# ---------------------------
# Formatting / explanation helpers
# ---------------------------
def format_steps(eq_objs: List[Eq], solutions: Any, unknown_syms: List[str]) -> str:
    """
    Create simple LaTeX/html steps describing how we solved.
    Returns an HTML string of paragraphs.
    """
    steps: List[str] = []
    for eq in eq_objs:
        steps.append(f"Equation: $$ {latex(eq.lhs)} = {latex(eq.rhs)} $$")

    if isinstance(solutions, dict):
        sol_str = ", ".join(f"{k} = {latex(v)}" for k, v in solutions.items())
        steps.append(f"Solution: $$ {sol_str} $$")
    elif isinstance(solutions, (list, tuple, set)):
        if len(solutions) == 0:
            steps.append("No solution found.")
        else:
            sol_str = ", ".join(latex(s) for s in solutions)
            if len(unknown_syms) == 1:
                steps.append(f"Solutions for {unknown_syms[0]}: $$ {sol_str} $$")
            else:
                steps.append(f"Solutions: $$ {sol_str} $$")
    else:
        steps.append(f"Solution: $$ {latex(solutions)} $$")

    steps.append("We verified the solution by substitution into the original equation(s).")
    html = "".join(f"<p>{s}</p>" for s in steps)
    return html


# ---------------------------
# Orchestration / metadata helpers
# ---------------------------
def normalize_requested_solver_from_job(job: Dict[str, Any]) -> str:
    """
    Return normalized requested solver label for the job (for observability).
    """
    code_info = job.get("simulated_codegen") or job.get("generated_code") or {}
    mode = code_info.get("mode") if isinstance(code_info, dict) else None
    if mode:
        return mode
    if job.get("requested_solver"):
        return job["requested_solver"]
    return "sympy_stub"


# ---------------------------
# High-level SymPy wrapper
# ---------------------------
def local_sympy_solve_from_question(question: str) -> Dict[str, Any]:
    """
    Convenience wrapper: parse question -> build eqs -> solve -> validate -> format result dict.

    Returns:
        {
          "answer": str(...),
          "structured": {"type":"algebra","solutions": str(...)},
          "steps_html": "...",
          "solved_by": "sympy_stub",
          "validation": {"ok": True/False, "method": "substitution"}
        }
    Raises ValueError if parsing fails.
    """
    # parse equations (attempt a couple heuristics)
    equations = extract_equations(question)
    if not equations:
        m = re.search(r"solve for ([a-zA-Z])[:\s]*(.*)", question, re.IGNORECASE)
        if m:
            var = m.group(1)
            expr = m.group(2).strip()
            if expr:
                equations = [(expr, "0")]

    if not equations:
        raise ValueError("Could not parse equation from question (simple parser).")

    eq_objs = build_sympy_eqs(equations)

    unk_names = detect_unknowns(equations)
    sympy_syms = symbols(" ".join(unk_names))
    if not isinstance(sympy_syms, (list, tuple)):
        sympy_syms = [sympy_syms]

    # solve using sympy (single equation vs system)
    from sympy import solve as sym_solve

    try:
        if len(eq_objs) == 1 and len(sympy_syms) == 1:
            sol = sym_solve(eq_objs[0], sympy_syms[0])
            solutions = sol
        else:
            sol = sym_solve(eq_objs, sympy_syms, dict=True)
            solutions = sol
    except Exception as e:
        # bubble up parse/solve errors with meaningful message
        raise ValueError(f"SymPy solve error: {e}")

    valid = validate_solution(eq_objs, solutions, sympy_syms)
    steps_html = format_steps(eq_objs, solutions, [str(s) for s in sympy_syms]) if valid else "<p>Solution found but validation failed.</p>"

    result = {
        "answer": str(solutions),
        "structured": {"type": "algebra", "solutions": str(solutions)},
        "steps_html": steps_html,
        "solved_by": "sympy_stub",
        "validation": {"ok": bool(valid), "method": "substitution"},
    }
    return result

# ---------------------------
# Helpers: unwrap & tolerant JSON extraction
# ---------------------------
def unwrap_markdown_code(s: str) -> str:
    """
    Remove triple-backtick fences and surrounding commentary from model outputs.
    If no fences, returns original stripped string.
    """
    if not s:
        return ""
    s.strip()
    # common fenced block pattern 
    m = re.search(r"```(?:python|py)?\n(.+?)\n```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # also strip single-line fenced content ```...```
    m2 = re.search(r"```(.+?)```", s, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    # remove leading ```python or ```json markers, trailing ```
    s = re.sub(r"^```(?:python|py|json)?\n?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\n?```$", "", s, flags=re.IGNORECASE).strip()
    return s

def extract_json_from_text_tolerant(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract JSON/dict from arbitrary text produced by model or executor.
    Returns parsed dict or None.
    """
    if not text:
        return None
    t = text.strip()
    # If text looks like a fenced block, unwrap first
    t_unwrapped = unwrap_markdown_code(t)
    # try direct parse
    try:
        return json.loads(t_unwrapped)
    except Exception:
        pass
    # fallback: find first {...} block
    m = re.search(r"(\{.*\})", t, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(1)
    # progressively try to trim trailing garbage to find valid JSON
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except Exception:
            continue
    # last attempt: try to eval as python literal safely (not recommended),
    # we avoid eval to be safe — return None
    return None

    

