# tests/test_workflow.py

"""
Full workflow test: user question -> parse -> codegen -> orchestrate -> explain -> Redis update
Mimics real container dev environment.
"""

import os
import json
import time
import redis
import orchestrator
import llm_wrapper

# -------------------- Configuration --------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
r = redis.from_url(REDIS_URL, decode_responses=True)

# -------------------- Sample Job --------------------
job_id = "workflow_test_job"
question = "Solve for x: 2*x + 3 = 11"

job_data = {
    "job_id": job_id,
    "question": question,
    "status": "pending",
    "created_at": time.time(),
}

# Submit job to Redis
r.set(f"job:{job_id}", json.dumps(job_data))
print(f"Submitted job {job_id} to Redis!")

# -------------------- Orchestrate Job --------------------
print("\n--- Running orchestrator (SymPy fallback) ---")
result_sympy = orchestrator.orchestrate_job(job_id, allow_docker_executor=False)
print(json.dumps(result_sympy, indent=2))

print("\n--- Running orchestrator (LLM/Docker path simulation) ---")
result_llm = orchestrator.orchestrate_job(job_id, allow_docker_executor=True)
print(json.dumps(result_llm, indent=2))

# -------------------- Explain Result --------------------
expl_sympy = llm_wrapper.explain_result(question, result_sympy)
expl_llm = llm_wrapper.explain_result(question, result_llm)

print("\n--- SymPy Explanation ---")
print(json.dumps(expl_sympy, indent=2))

print("\n--- LLM Explanation ---")
print(json.dumps(expl_llm, indent=2))

# -------------------- Check Updated Job in Redis --------------------
updated_job = json.loads(r.get(f"job:{job_id}"))
print("\n--- Updated Job Record in Redis ---")
print(json.dumps(updated_job, indent=2))

print("\n✅ Workflow test completed successfully!")



