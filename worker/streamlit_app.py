# worker/streamlit_app.py
"""
Simple Streamlit UI for testing the PiBot pipeline end-to-end:
 - parse question
 - generate code
 - run (SymPy fallback OR Docker executor)
 - explain (LLM explainer)
This app intentionally uses the same modules as the worker so it
mimics actual runtime behavior inside the worker container.
"""

from __future__ import annotations
import json
import textwrap
import time
import traceback

import streamlit as st

st.set_page_config(page_title="PiBot E2E Tester", layout="wide")

# Try to import the worker modules; show friendly error if not available
try:
    import llm_wrapper 
    import utils
    import safety
    import orchestrator
except Exception as e:
    st.error("Could not import worker modules. Run this app inside the worker container or ensure PYTHONPATH includes the worker package.")
    st.exception(e)
    st.stop()

st.title("PiBot - E2E Workflow Tester")

with st.sidebar:
    st.markdown("## Run Options")
    run_mode = st.radio(
        "Executor",
        ("Local SymPy (safe, fast)", "Docker Executor (requires docker + executor image)"),
    )
    enable_llm = st.checkbox("Use LLM wrapper (parse/codegen/explain)", value=True)
    allow_docker = st.checkbox("Allow docker executor (when chosen)", value=False)
    st.markdown("---")
    st.markdown("**Notes**")
    st.write(
        "• Docker executor requires `docker` available inside the worker container and the executor image built.\n"
        "• Safety scan is run on generated code before any execution.\n"
        "• Use Docker only on trusted host."
    )

# Main input
st.header("1. Enter a math question")
question = st.text_area("Question", value="Solve for x: 2*x + 3 = 11", height=120)

col1, col2, col3 = st.columns([1, 1, 1])

if col1.button("Parse"):
    st.session_state["parsed_at"] = time.time()
    try:
        parsed = llm_wrapper.parse_text_to_json(question) if enable_llm else {
            "raw_question": question,
            "equations": utils.extract_equations(question),
            "unknowns": utils.detect_unknowns(utils.extract_equations(question) or []),
            "goal": "solve",
            "notes": "heuristic"
        }
        st.session_state["parsed"] = parsed
        st.success("Parsed")
    except Exception as e:
        st.error("Parser error")
        st.exception(e)

if col2.button("Generate Code"):
    st.session_state["codegen_at"] = time.time()
    try:
        structured = st.session_state.get("parsed") or llm_wrapper.parse_text_to_json(question)
        code_out = llm_wrapper.generate_python_from_json(structured)
        st.session_state["codegen"] = code_out
        st.success(f"Codegen done (mode={code_out.get('mode')})")
    except Exception as e:
        st.error("Codegen error")
        st.exception(e)

if col3.button("Run end-to-end (parse→codegen→exec→explain)"):
    st.session_state["workflow_at"] = time.time()
    try:
        # Parse (prefer stored)
        structured = st.session_state.get("parsed") or (llm_wrapper.parse_text_to_json(question) if enable_llm else {"raw_question": question, "equations": utils.extract_equations(question)})
        st.session_state["parsed"] = structured

        # Codegen
        code_out = st.session_state.get("codegen") or llm_wrapper.generate_python_from_json(structured) if enable_llm else {"mode": "sympy_stub", "code": None}
        st.session_state["codegen"] = code_out

        # Safety scan
        code_text = code_out.get("code") or ""
        safe_ok, issues = (True, [])
        if code_text:
            safe_ok, issues = safety.is_code_safe(code_text)
        st.session_state["safety"] = {"ok": safe_ok, "issues": issues}

        # Execute
        exec_result = None
        exec_meta = {"path": None}
        if code_out.get("mode") == "llm_codegen" and code_text and safe_ok:
            if run_mode.startswith("Docker") and allow_docker:
                # Use orchestrator docker executor
                try:
                    exec_out = orchestrator.executor_run_docker_script(code_text, job_id=f"streamlit-{int(time.time())}")
                    # try parse stdout as json
                    try:
                        parsed_out = json.loads(exec_out.get("stdout") or "{}")
                    except Exception:
                        parsed_out = {"stdout": exec_out.get("stdout"), "stderr": exec_out.get("stderr"), "exit_code": exec_out.get("exit_code")}
                    exec_result = parsed_out
                    exec_meta["path"] = "docker_executor"
                except Exception as e:
                    exec_result = {"error": str(e)}
                    exec_meta["path"] = "docker_executor_error"
            else:
                # fallback: don't execute unknown code; run local sympy instead
                try:
                    exec_result = utils.local_sympy_solve_from_question(question)
                    exec_meta["path"] = "sympy_fallback_from_llm_codegen"
                except Exception as e:
                    exec_result = {"error": str(e)}
                    exec_meta["path"] = "sympy_error"
        else:
            # sympy path
            try:
                exec_result = utils.local_sympy_solve_from_question(question)
                exec_meta["path"] = "sympy"
            except Exception as e:
                exec_result = {"error": str(e)}
                exec_meta["path"] = "sympy_error"

        st.session_state["exec_result"] = exec_result
        st.session_state["exec_meta"] = exec_meta

        # Explain
        try:
            # pass generated code if available for richer explanation
            gen_code_for_explain = code_text if code_text else None
            expl = llm_wrapper.explain_result(question, exec_result, generated_code=gen_code_for_explain) if enable_llm else {"steps_html": exec_result.get("steps_html") or "<p>No explanation (LLM disabled)</p>", "confidence": 0.0}
            st.session_state["explain"] = expl
        except Exception as e:
            st.session_state["explain"] = {"steps_html": f"<p>Explainer error: {e}</p>", "confidence": 0.0}

        st.success("Workflow finished")
    except Exception as e:
        st.error("Workflow failed")
        st.exception(traceback.format_exc())

st.markdown("---")

# Show panels
parsed = st.session_state.get("parsed")
codegen = st.session_state.get("codegen")
safety_info = st.session_state.get("safety")
exec_result = st.session_state.get("exec_result")
expl = st.session_state.get("explain")
exec_meta = st.session_state.get("exec_meta")

st.header("Parse result")
if parsed:
    st.json(parsed)
else:
    st.info("No parse data yet. Click Parse or Run end-to-end.")

st.header("Generated code")
if codegen:
    st.write("Mode:", codegen.get("mode"))
    code_to_show = codegen.get("code") or "<no code produced (sympy fallback)>"
    st.code(code_to_show, language="python")
    if safety_info:
        if not safety_info["ok"]:
            st.error("Safety issues detected:")
            for it in safety_info["issues"]:
                st.write("-", it)
        else:
            st.success("Safety: OK")
else:
    st.info("No code generated yet.")

st.header("Execution result")
if exec_result:
    # pretty print if dict-like
    if isinstance(exec_result, (dict, list)):
        st.json(exec_result)
    else:
        st.text(str(exec_result))
    st.write("Exec meta:", exec_meta)
else:
    st.info("Not executed yet.")

import streamlit.components.v1 as components

st.header("Explanation")
if expl:
    steps_html = expl.get("steps_html", "")
    st.markdown("**Confidence:** " + str(expl.get("confidence", 0.0)))

    # Clean <html> tags and inject styling for dark mode
    clean_html = steps_html.replace('<html>', '').replace('</html>', '')
    styled_html = f"""
    <div style="
        color: #f0f0f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        font-size: 1rem;
    ">
    {clean_html}
    </div>
    """

    components.html(styled_html, scrolling=True, height=650)

    # Optional: debug raw data
    with st.expander("🔍 Raw explanation (debug)"):
        st.json(expl)
else:
    st.info("No explanation yet.")

st.markdown("---")
st.caption("Streamlit E2E tester — does not auto-run untrusted code unless Docker executor is enabled.")
