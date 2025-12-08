#!/usr/bin/env python3
# test_e2e.py
"""
End-to-end workflow tester for PIBOT worker.

Usage (inside worker container):
  python test_e2e.py --mode sympy --job_id test_sympy --question "Solve for x: 2*x + 3 = 11"
  python test_e2e.py --mode docker --job_id test_docker --question "Solve for x: 2*x + 3 = 11"

Notes:
- This script writes job:{job_id} into Redis and invokes the orchestrator and/or executor helpers.
- For docker mode you must have the docker socket mounted into the worker container
  (e.g. -v /var/run/docker.sock:/var/run/docker.sock) and the executor image built (pibot-executor:latest).
"""

import os
import json
import time
import argparse
import redis

# Lazy imports for environment within container
try:
    import orchestrator
    import llm_wrapper
    import utils
except Exception:
    # If orchestrator/llm_wrapper are importable as top-level worker modules, try that
    import importlib
    orchestrator = importlib.import_module("worker.orchestrator")
    llm_wrapper = importlib.import_module("worker.llm_wrapper")
    utils = importlib.import_module("worker.utils")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
r = redis.from_url(REDIS_URL, decode_responses=True)

def create_job_record(job_id: str, question: str):
    job = {
        "job_id": job_id,
        "question": question,
        "status": "pending",
        "created_at": time.time(),
    }
    r.set(f"job:{job_id}", json.dumps(job))
    print(f"Submitted job {job_id} to Redis.")

def run_sympy_flow(job_id: str):
    print("\n--- Running orchestrator (SymPy fallback) ---")
    res = orchestrator.orchestrate_job(job_id, allow_docker_executor=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    updated = json.loads(r.get(f"job:{job_id}"))
    print("\n--- Updated job in Redis ---")
    print(json.dumps(updated, indent=2, ensure_ascii=False))

def run_docker_flow(job_id: str, question: str):
    """
    Simulate LLM codegen by producing a safe python snippet that prints JSON,
    then run it in the executor container via orchestrator.executor_run_docker_script().
    Finally call explainer (llm_wrapper.explain_result) with question, exec_result and the code.
    """
    print("\n--- Running executor (docker) simulation ---")

    # Simple safe generated code that uses sympy to solve typical linear/quadratic eqns.
    # It must print a single JSON object to stdout.
    # NOTE: Keep code minimal and safe (no network, no os/sys)
    generated_code = f"""
import json
import sympy

# This is a generated safe stub for: {question}
# Try to parse simple equation using sympy with implicit multiplication tolerant parsing.
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
transformations = standard_transformations + (implicit_multiplication_application,)

# Very naive extraction: attempt to find text around '='
q = {json.dumps(question)}
if '=' in q:
    lhs, rhs = q.split('=', 1)
else:
    lhs, rhs = q, '0'
lhs = lhs.strip().replace('Solve for x:', '').strip()
rhs = rhs.strip()
try:
    x = sympy.symbols('x')
    lhs_expr = parse_expr(lhs, transformations=transformations, evaluate=True)
    rhs_expr = parse_expr(rhs, transformations=transformations, evaluate=True)
    eq = sympy.Eq(lhs_expr, rhs_expr)
    sol = sympy.solve(eq, x)
    # normalize sol to JSON-friendly types
    out_solutions = []
    for s in sol:
        try:
            out_solutions.append(float(s))
        except Exception:
            out_solutions.append(str(s))
    result = {{
        "answer": str(out_solutions),
        "structured": {{"type":"algebra","solutions": out_solutions}},
        "steps_html": "<p>Generated-executor result</p>",
        "solved_by": "generated_executor",
    }}
except Exception as e:
    result = {{"error": "executor_error", "message": str(e)}}

print(json.dumps(result))
"""

    # Run the executor via orchestrator helper (this will write a temp dir and call docker)
    try:
        exec_out = orchestrator.executor_run_docker_script(generated_code, job_id, timeout=10)
    except Exception as e:
        print("Executor run raised exception:", str(e))
        return

    print("\nExecutor raw output:")
    print("stdout:", exec_out.get("stdout")[:1000])
    print("stderr:", exec_out.get("stderr")[:1000])
    print("exit_code:", exec_out.get("exit_code"))

    # try to parse JSON stdout
    parsed = {}
    try:
        parsed = json.loads(exec_out.get("stdout") or "{}")
    except Exception as e:
        parsed = {"error": "parse_stdout_failed", "stdout": exec_out.get("stdout"), "stderr": exec_out.get("stderr"), "parse_exc": str(e)}

    # Ask the explainer to produce steps (it accepts generated_code optional)
    try:
        expl = llm_wrapper.explain_result(question, parsed, generated_code)
    except Exception as e:
        print("Explainer raised:", e)
        expl = {"steps_html": "", "confidence": 0.0}

    # Compose final result
    final_result = parsed.copy() if isinstance(parsed, dict) else {"answer": str(parsed)}
    if expl and expl.get("steps_html"):
        final_result["steps_html"] = expl["steps_html"]
        final_result.setdefault("explain_meta", {})["from_llm"] = True
    final_result.setdefault("solved_by", "executor_container")
    final_result.setdefault("execution_path", "docker_executor")
    final_result.setdefault("validation", {"ok": False})

    # Persist to Redis job record
    existing = r.get(f"job:{job_id}")
    if existing:
        job = json.loads(existing)
    else:
        job = {"job_id": job_id, "question": question}
    job.update({"status": "done", "finished_at": time.time(), "result": final_result})
    r.set(f"job:{job_id}", json.dumps(job))
    print("\n--- Final composed result ---")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    print("\n--- Job stored into Redis ---")
    print(json.dumps(job, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sympy", "docker"], default="sympy")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    create_job_record(args.job_id, args.question)

    if args.mode == "sympy":
        run_sympy_flow(args.job_id)
    else:
        run_docker_flow(args.job_id, args.question)


if __name__ == "__main__":
    main()
