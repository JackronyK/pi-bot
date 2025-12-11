# worker/streamlit_app.py
"""

Streamlit UI for PiBot E2E Testing Platform
============================================
Professional API-driven interface for testing the complete pipeline:
- File uploads with OCR
- Question parsing
- Code generation
- Execution
- Explanation generation

"""


from __future__ import annotations
import json
import os
import textwrap
import time
import traceback
from typing import Dict, Optional

import streamlit as st
import requests
import streamlit.components.v1 as components

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PiBot E2E Testing Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONFIGURATION & SIDEBAR
# ============================================================================
DEFAULT_API = os.environ.get("API_BASE", "http://api:8000")

with st.sidebar:
    st.markdown('### ⚙️ Configuration')

    #API Configuration
    st.markdown('### API Settings')
    API_BASE = st.text_input(
        "API base URL",
        value=DEFAULT_API,
        help="Use 'http://api:8000' for Docker Compose or 'http://localhost:8000' for local"
    )

    # Connection test
    if st.button("🔌 Test Connection"):
        try:
            response = requests.get(f"{API_BASE.rstrip('/')}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connection successful!")
            else:
                st.warning(f"⚠️ Server responded with status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.markdown("---")

    # Execution Settings
    st.markdown("#### Execution Settings")
    allow_docker_execution = st.checkbox(
        "Enable Docker Executor",
        value=True,
        help="Run generated code in isolated Docker container (recommended)"
    )
    
    use_llm_wrapper = st.checkbox(
        "Use LLM Wrapper",
        value=True,
        help="Enable AI-powered code generation and explanation"
    )
    
    execution_timeout = st.slider(
        "Execution Timeout (seconds)",
        min_value=10,
        max_value=180,
        value=30,
        help="Maximum time allowed for code execution"
    )
    
    st.markdown("---")


    # Session Management
    st.markdown("#### Session Management")
    if st.button("🗑️ Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.caption("PiBot E2E Testing Platform v1.0")


# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================
def call_api_endpoint(
    endpoint: str,
    json_payload: Optional[Dict] = None,
    files=None,
    timeout: int = 30
) -> Dict:
    """
    Make API call with comprehensive error handling.
    
    Args:
        endpoint: API endpoint path (e.g., '/solve/parse')
        json_payload: JSON data to send
        files: Files to upload
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary containing response data or error information
    """
    url = f"{API_BASE.rstrip('/')}{endpoint}"
    try:
        if files is not None:
            response = requests.post(url, files=files, timeout=timeout)
        else:
            response = requests.post(url, json=json_payload or {}, timeout=timeout)
        
        response.raise_for_status()
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw_text": response.text}
            
    except requests.exceptions.Timeout:
        return {
            "_error": "Request timeout",
            "message": f"Request exceeded {timeout} seconds"
        }
    except requests.exceptions.ConnectionError:
        return {
            "_error": "Connection error",
            "message": "Could not connect to API server"
        }
    except requests.exceptions.HTTPError as e:
        return {
            "_error": f"HTTP {e.response.status_code}",
            "message": str(e),
            "details": e.response.text
        }
    except Exception as e:
        return {
            "_error": "Unexpected error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

def display_error(error_data: Dict):
    """Display formatted error message."""
    st.error(f"**Error:** {error_data.get('_error', 'Unknown error')}")
    if error_data.get('message'):
        st.error(f"**Message:** {error_data['message']}")
    if error_data.get('details'):
        with st.expander("📋 Error Details"):
            st.code(error_data['details'])

# ============================================================================
# MAIN APPLICATION
# ============================================================================
st.markdown('<div class="main-header">🤖 PiBot E2E Testing Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comprehensive testing interface for question parsing, code generation, and execution</div>', unsafe_allow_html=True)

# ============================================================================
# SECTION 1: INPUT
# ============================================================================
st.markdown('<div class="section-header">📝 Step 1: Input Question</div>', unsafe_allow_html=True)

col_input, col_actions = st.columns([2, 1])

with col_input:
    input_mode = st.radio(
        "Select Input Method",
        ["✍️ Type Question", "📄 Upload File (Image/PDF)"],
        horizontal=True
    )

    question_text = ""
    extracted_text = ""

    if "Type" in input_mode:
        question_text = st.text_area(
            "Enter your question",
            value = "Solve for x: x^2 - 7x + 10 = 0",
            height=180,
            placeholder="Type your mathematical question here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload image or PDF containing question",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supported formats: PDF, PNG, JPG, JPEG"
        )

        if uploaded_file:
            with st.spinner("🔄 Processing file and extracting text..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = call_api_endpoint("/uploads/submit-file", files=files, timeout=60)

            if response.get("_error"):
                display_error(response)
            else:
                result = response.get("result", {})
                extracted_text = result.get("extracted_text", "")

                st.success("✅ OCR completed successfully!")

                with st.expander("📄 View Extracted Text"):
                    st.code(extracted_text, language=None)

                question_text = st.text_area(
                    "Review and edit extracted text",
                    value=extracted_text,
                    height=200,
                    help="Verify and correct any OCR errors"
                )

with col_actions:
    st.markdown("### 🚀 Quick Actions")
    
    # Parse Buttonn
    if st.button("🔍 Parse Question", use_container_width=True):
        if not question_text.strip():
            st.warning("⚠️ Please provide a question first")
        else:
            with st.spinner("Parsing question..."):
                response = call_api_endpoint("/solve/parse", {"question": question_text})

            if response.get("_error"):
                display_error(response)
            else:
                st.session_state["parsed_result"] = response.get("parsed", response)
                st.success("✅ Question parsed successfully!")

    # Generate Code Button
    if st.button("⚙️ Generate Code", use_container_width=True):
        parsed_data = st.session_state.get("parsed_result")
        
        if not parsed_data:
            with st.spinner("Parsing question first..."):
                parse_response = call_api_endpoint("/solve/parse", {"question": question_text})
            
            if parse_response.get("_error"):
                display_error(parse_response)
            else:
                parsed_data = parse_response.get("parsed", parse_response)
                st.session_state["parsed_result"] = parsed_data
        
        if parsed_data:
            with st.spinner("Generating Python code..."):
                response = call_api_endpoint("/solve/codegen", {"structured": parsed_data})
            
            if response.get("_error"):
                display_error(response)
            else:
                st.session_state["codegen_result"] = response
                st.success("✅ Code generated successfully!")

    # Execute Code Button
    if st.button("▶️ Execute Code", use_container_width=True):
        codegen_result = st.session_state.get("codegen_result", {})
        code_text = codegen_result.get('code', '')
        
        if not code_text:
            st.warning("⚠️ No generated code available. Generate code first!")
        else:
            payload = {
                "generated_code": code_text,  # ✅ FIXED: Send only the code string
                "use_docker_executor": allow_docker_execution,
                "timeout_seconds": execution_timeout
            }
            
            with st.spinner("Executing code..."):
                response = call_api_endpoint("/solve/exec-code", payload, timeout=execution_timeout + 10)
            
            if response.get("_error"):
                display_error(response)
            else:
                st.session_state["execution_result"] = response
                st.success("✅ Code executed successfully!")
    
    # Run Full Pipeline Button
    if st.button("🎯 Run Complete Pipeline", type="primary", use_container_width=True):
        if not question_text.strip():
            st.warning("⚠️ Please provide a question first")
        else:
            payload = {
                "question": question_text,
                "use_docker_executor": allow_docker_execution,
                "use_llm": use_llm_wrapper
            }
            
            with st.spinner("Running complete pipeline... This may take a minute."):
                response = call_api_endpoint("/solve/full", payload, timeout=180)
            
            if response.get("_error"):
                display_error(response)
            else:
                st.session_state["pipeline_result"] = response
                st.session_state["execution_result"] = response
                st.success("✅ Pipeline completed successfully!")

st.markdown("---")


# ============================================================================
# SECTION 2: PARSED RESULT
# ============================================================================
st.markdown('<div class="section-header">🔍 Step 2: Parsed Structure</div>', unsafe_allow_html=True)

parsed_result = st.session_state.get("parsed_result")
if parsed_result:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.json(parsed_result, expanded=True)
    with col2:
        st.metric("Status", "✅ Parsed")
        if st.button("🔄 Re-parse"):
            if question_text:
                with st.spinner("Re-parsing..."):
                    response = call_api_endpoint("/solve/parse", {"question": question_text})
                if not response.get("_error"):
                    st.session_state["parsed_result"] = response.get("parsed", response)
                    st.reruun()
else:
    st.info("ℹ️ No parsed result yet. Click **Parse Question** to begin.")

st.markdown("---")

# ============================================================================
# SECTION 3: GENERATED CODE
# ============================================================================
st.markdown('<div class="section-header">⚙️ Step 3: Generated Code</div>', unsafe_allow_html=True)

codegen_result = st.session_state.get("codegen_result")
if codegen_result:
    col1, col2 = st.columns([4, 1])

    with col1:
        mode = codegen_result.get("mode", "unknown")
        st.info(f"**Generation Mode:** `{mode}`")
        
        code_text = codegen_result.get("code", "# No code generated")
        st.code(code_text, language="python", line_numbers=True)
        
        # Download button
        st.download_button(
            label="📥 Download Code",
            data=code_text,
            file_name="generated_code.py",
            mime="text/x-python"
        )

    with col2:
        st.metric("Status", "✅ Generated")
        st.metric("Lines", len(code_text.split('\n')))
        
        if st.button("📋 Copy Code"):
            st.code(code_text)
            st.success("Code displayed above - use your browser's copy function")
else:
    st.info("ℹ️ No generated code yet. Click **Generate Code** to create executable Python code.")

st.markdown("---")

# ============================================================================
# SECTION 4: EXECUTION RESULT
# ============================================================================
st.markdown('<div class="section-header">▶️ Step 4: Execution Result</div>', unsafe_allow_html=True)

execution_result = st.session_state.get("execution_result") or st.session_state.get("pipeline_result")

if execution_result:
    # Display metadata
    if isinstance(execution_result, dict):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            solver = execution_result.get("requested_solver") or execution_result.get("solved_by", "N/A")
            st.metric("Solver Used", solver)
        
        with col2:
            exec_path = execution_result.get("execution_path", "N/A")
            st.metric("Execution Path", exec_path)
        
        with col3:
            exit_code = execution_result.get("exit_code", execution_result.get("status", "N/A"))
            st.metric("Exit Code", exit_code)
        
        # Display full result
        with st.expander("📊 Full Execution Details", expanded=True):
            st.json(execution_result, expanded=False)
        
        # Extract and display key results
        if "stdout" in execution_result:
            st.markdown("**Standard Output:**")
            st.code(execution_result["stdout"], language="json")
        
        if "stderr" in execution_result and execution_result["stderr"]:
            st.markdown("**Standard Error:**")
            st.error(execution_result["stderr"])
    else:
        st.text(str(execution_result))
else:
    st.info("ℹ️ No execution result yet. Click **Execute Code** or **Run Complete Pipeline**.")

st.markdown("---")

# ============================================================================
# SECTION 5: EXPLANATION
# ============================================================================
st.markdown('<div class="section-header">💡 Step 5: Solution Explanation</div>', unsafe_allow_html=True)

explanation = st.session_state.get("explanation_result")

# Try to get explanation from pipeline result if not explicitly set
if not explanation and execution_result:
    steps_html = execution_result.get("steps_html")
    if steps_html:
        explanation = {
            "steps_html": steps_html,
            "confidence": execution_result.get("validation", {}).get("ok", 0.0)
        }

if explanation:
    # Display confidence score
    confidence = explanation.get("confidence")
    if confidence is not None:
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Confidence Score", f"{confidence_pct:.1f}%")
    
    # Display HTML explanation
    steps_html = explanation.get("steps_html", "")
    
    if steps_html:
        # Clean and style the HTML
        cleaned_html = steps_html.replace("<html>", "").replace("</html>", "").strip()
        
        styled_html = f"""
        <div style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.8;
            padding: 20px;
            color: #1a1a1a;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            {cleaned_html}
        </div>
        """
        
        components.html(styled_html, height=600, scrolling=True)
    
    # Debug information
    with st.expander("🔧 Debug: Raw Explanation Data"):
        st.json(explanation)
else:
    st.info("ℹ️ No explanation yet. Run the complete pipeline to generate a detailed explanation.")
    
    # Offer to generate explanation if we have execution result
    if execution_result and question_text:
        if st.button("📝 Generate Explanation Now"):
            payload = {
                "question": question_text,
                "result": execution_result,
                "generated_code": codegen_result.get("code") if codegen_result else None
            }
            
            with st.spinner("Generating explanation..."):
                response = call_api_endpoint("/solve/explain", payload)
            
            if response.get("_error"):
                display_error(response)
            else:
                st.session_state["explanation_result"] = response
                st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>PiBot E2E Testing Platform</strong></p>
    <p>This interface communicates with your API backend. All code execution happens server-side in isolated containers.</p>
    <p style='font-size: 0.9em;'>Made with ❤️ using Streamlit</p>
    <p>@Jackrony 2025</p>
</div>
""", unsafe_allow_html=True)
