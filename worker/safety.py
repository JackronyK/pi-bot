# worker/ safety.py
"""
Static safety checker for generated Python code.

Provides simple AST-based checks to detect forbidden imports and dangerous calls/names
that should not appear in generated solver code. This is intended as a static pre-execution
sanity gate for LLM-generated code.

Usage:
    from worker.safety import scan_code, is_code_safe

    issues = scan_code(code_str)
    if issues:
        # reject or request regeneration

CLI:
    python worker/safety.py path/to/generated_code.py

Notes / limitations:
- This is conservative: it errs on the side of flagging suspicious constructs.
- It doesn't guarantee absolute safety — runtime sandboxing is still required.
"""

from __future__ import annotations

import ast
import sys
from typing import List, Tuple

# A conservative list of forbidden top-level imports (modules or names)
FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "pathlib",
    "builtins",
}

# Forbidden function names when called directly
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
}

# Forbidden attribute bases that indicate dangerous actions (e.g., os.system)
FORBIDDEN_ATTR_BASES = {
    "os",
    "subprocess",
    "socket",
    "shutil",
}

# Forbidden attribute names often used to execute or access system resources
FORBIDDEN_ATTR_NAMES = {
    "system",
    "popen",
    "Popen",
    "spawn",
    "connect",
    "recv",
    "send",
    "open",  # e.g. module.open
}


def _get_full_attr_name(node: ast.Attribute) -> str:
    """Return dotted name for attribute node, e.g. subprocess.Popen -> 'subprocess.Popen'"""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)

class SafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in FORBIDDEN_IMPORTS:
                self.issues.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        base = module.split(".")[0] if module else ""
        if base in FORBIDDEN_IMPORTS:
            self.issues.append(f"Forbidden import from '{module}' at line {node.lineno}")
        for alias in node.names:
            if alias.name in FORBIDDEN_IMPORTS:
                self.issues.append(f"Forbidden import '{alias.name}' from {module} at line {node.lineno}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for direct forbidden calls like eval(...), exec(...), open(...)
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in FORBIDDEN_CALLS:
                self.issues.append(f"Forbidden call to '{fname}' at line {node.lineno}")
        # Check attribute calls like os.system(), subprocess.Popen()
        elif isinstance(node.func, ast.Attribute):
            full = _get_full_attr_name(node.func)
            # Check if base is forbidden
            base = full.split(".")[0]
            if base in FORBIDDEN_ATTR_BASES:
                self.issues.append(f"Forbidden attribute call '{full}' at line {node.lineno}")
            # Check if attribute name itself is suspicious
            attr_name = node.func.attr
            if attr_name in FORBIDDEN_ATTR_NAMES:
                self.issues.append(f"Suspicious attribute '{attr_name}' used in call at line {node.lineno} (full: {full})")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Flag attribute access that looks like os.system (even if not called)
        try:
            full = _get_full_attr_name(node)
            base = full.split(".")[0]
            if base in FORBIDDEN_ATTR_BASES:
                # attribute access on forbidden base
                self.issues.append(f"Attribute access on forbidden base '{full}' at line {node.lineno}")
            if node.attr in FORBIDDEN_ATTR_NAMES:
                self.issues.append(f"Suspicious attribute '{node.attr}' accessed at line {node.lineno} (full: {full})")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Detect __import__ name usage
        if node.id == "__import__":
            self.issues.append(f"Use of __import__ at line {node.lineno}")
        self.generic_visit(node)


def scan_code(code: str) -> List[str]:
    """Scan code string for suspicious constructs. Returns list of issue messages (empty if none)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError while parsing code: {e}"]

    v = SafetyVisitor()
    v.visit(tree)
    return v.issues


def is_code_safe(code: str) -> Tuple[bool, List[str]]:
    issues = scan_code(code)
    return (len(issues) == 0, issues)

# CLI support
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python worker/safety.py path/to/generated_code.py")
        sys.exit(2)
    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf8") as f:
            src = f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        sys.exit(1)

    ok, issues = is_code_safe(src)
    if ok:
        print("OK: No safety issues detected.")
        sys.exit(0)
    else:
        print("Safety issues detected:")
        for it in issues:
            print(" - ", it)
        sys.exit(3)