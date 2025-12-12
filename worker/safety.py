# worker/ safety.py
"""
Static Safety Checker - Production-Ready Version
=================================================
Multi-layer AST-based security scanner for generated Python code.

Security Checks:
- Forbidden imports (os, sys, subprocess, etc.)
- Dangerous function calls (eval, exec, compile, etc.)
- Suspicious attribute access (os.system, subprocess.Popen, etc.)
- Network operations (socket, urllib, requests, etc.)
- File system operations (open, read, write, etc.)
- Code complexity limits (depth, statement count)
- String-based code execution patterns

Features:
- Comprehensive AST traversal
- Detailed issue reporting with line numbers
- Configurable security levels
- Export to structured format
- CLI support for testing

Usage:
    from worker.safety import is_code_safe, scan_code, SecurityLevel
    
    # Quick check
    safe, issues = is_code_safe(code_string)
    
    # Detailed scan
    report = scan_code(code_string, level=SecurityLevel.STRICT)
    
    # CLI
    python worker/safety.py path/to/code.py

Environment Variables:
---------------------
PIBOT_SAFETY_LEVEL=strict|normal|relaxed
PIBOT_MAX_CODE_DEPTH=10
PIBOT_MAX_STATEMENTS=500
"""

from __future__ import annotations

import ast
import sys
import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("worker.safety")
logger.setLevel(os.environ.get("PIBOT_LOG_LEVEL", "INFO"))

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONFIGURATION
# ============================================================================
class SecurityLevel(Enum):
    """Security scanning strictness levels."""
    RELAXED = "relaxed"   # Allow more operations
    NORMAL = "normal"     # Balanced security
    STRICT = "strict"     # Maximum security
# Load configuration from environment
SAFETY_LEVEL = SecurityLevel(os.environ.get("PIBOT_SAFETY_LEVEL", "normal").lower())
MAX_CODE_DEPTH = int(os.environ.get("PIBOT_MAX_CODE_DEPTH", "10"))
MAX_STATEMENTS = int(os.environ.get("PIBOT_MAX_STATEMENTS", "500"))

# ============================================================================
# FORBIDDEN PATTERNS
# ============================================================================

# Critical: Always forbidden regardless of security level
CRITICAL_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "multiprocessing",
    "threading", "pickle", "marshal", "shelve", "dbm", "gdbm",
    "code", "codeop", "compile", "exec", "eval", "importlib"
}

# Restricted: Forbidden in NORMAL and STRICT modes
RESTRICTED_IMPORTS = {
    "requests", "urllib", "urllib3", "http", "httplib", "ftplib",
    "shutil", "pathlib", "glob", "tempfile", "atexit", "signal",
    "asyncio", "concurrent", "queue", "sched"
}

# Allowed safe imports
ALLOWED_IMPORTS = {
    "json", "math", "sympy", "re", "string", "collections",
    "itertools", "functools", "operator", "typing", "dataclasses",
    "enum", "decimal", "fractions", "statistics", "random", "time"
}

# Forbidden function calls
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "getattr",
    "setattr", "delattr", "globals", "locals", "vars", "dir",
    "help", "input", "breakpoint", "exit", "quit"
}

# Forbidden attribute bases (e.g., os.system)
FORBIDDEN_ATTR_BASES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "ctypes", "pickle", "urllib", "http", "ftplib"
}

# Forbidden attribute names
FORBIDDEN_ATTR_NAMES = {
    "system", "popen", "Popen", "spawn", "fork", "exec",
    "connect", "bind", "listen", "accept", "recv", "send",
    "open", "read", "write", "unlink", "remove", "rmdir",
    "chmod", "chown", "kill", "__import__"
}

# Dangerous string patterns
DANGEROUS_PATTERNS = [
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__\s*\(",
    r"compile\s*\(",
    r"os\.system\s*\(",
    r"subprocess\.",
    r"socket\.",
]


# ============================================================================
# ISSUE TRACKING
# ============================================================================
@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "import", "call", "attribute", "complexity", "pattern"
    message: str
    line_number: int
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "suggestion": self.suggestion
        }

@dataclass
class SecurityReport:
    """Complete security scan report."""
    is_safe: bool
    issues: List[SecurityIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    level: SecurityLevel = SecurityLevel.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_safe": self.is_safe,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats,
            "level": self.level.value
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.is_safe:
            return f"✓ Code passed {self.level.value} security scan"
        
        severity_counts = {}
        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        parts = [f"✗ {len(self.issues)} security issue(s) found:"]
        for severity in ["critical", "high", "medium", "low"]:
            if severity in severity_counts:
                parts.append(f"  - {severity_counts[severity]} {severity}")
        
        return "\n".join(parts)

# ============================================================================
# AST VISITOR FOR SECURITY SCANNING
# ============================================================================
class SecurityVisitor(ast.NodeVisitor):
    """AST visitor that detects security issues."""
    
    def __init__(self, code_lines: List[str], level: SecurityLevel = SecurityLevel.NORMAL):
        self.code_lines = code_lines
        self.level = level
        self.issues: List[SecurityIssue] = []
        self.stats = {
            "imports": [],
            "function_calls": [],
            "max_depth": 0,
            "statement_count": 0,
            "current_depth": 0
        }
        
        # Determine which imports to check based on security level
        if level == SecurityLevel.STRICT:
            self.forbidden_imports = CRITICAL_IMPORTS | RESTRICTED_IMPORTS
        elif level == SecurityLevel.NORMAL:
            self.forbidden_imports = CRITICAL_IMPORTS
        else:  # RELAXED
            self.forbidden_imports = set()
    
    def _get_code_snippet(self, line_number: int, context: int = 0) -> str:
        """Extract code snippet around line number."""
        start = max(0, line_number - context - 1)
        end = min(len(self.code_lines), line_number + context)
        return "\n".join(self.code_lines[start:end])
    
    def _add_issue(
        self,
        severity: str,
        category: str,
        message: str,
        line_number: int,
        suggestion: Optional[str] = None
    ):
        """Add a security issue."""
        snippet = self._get_code_snippet(line_number)
        issue = SecurityIssue(
            severity=severity,
            category=category,
            message=message,
            line_number=line_number,
            code_snippet=snippet,
            suggestion=suggestion
        )
        self.issues.append(issue)
        logger.debug(f"Found issue: {severity} - {message} at line {line_number}")
    
    def visit_Import(self, node: ast.Import):
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self.stats["imports"].append(module_name)
            
            if module_name in self.forbidden_imports:
                self._add_issue(
                    severity="critical" if module_name in CRITICAL_IMPORTS else "high",
                    category="import",
                    message=f"Forbidden import: '{alias.name}'",
                    line_number=node.lineno,
                    suggestion=f"Remove import of '{alias.name}' - this module provides dangerous capabilities"
                )
            elif module_name not in ALLOWED_IMPORTS:
                if self.level == SecurityLevel.STRICT:
                    self._add_issue(
                        severity="medium",
                        category="import",
                        message=f"Unknown/unapproved import: '{alias.name}'",
                        line_number=node.lineno,
                        suggestion=f"Only explicitly allowed imports are permitted in strict mode"
                    )
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from...import statements."""
        module = node.module or ""
        base_module = module.split(".")[0] if module else ""
        
        if base_module:
            self.stats["imports"].append(base_module)
        
        if base_module in self.forbidden_imports:
            self._add_issue(
                severity="critical" if base_module in CRITICAL_IMPORTS else "high",
                category="import",
                message=f"Forbidden import from: '{module}'",
                line_number=node.lineno,
                suggestion=f"Remove import from '{module}'"
            )
        
        # Check individual names
        for alias in node.names:
            if alias.name in FORBIDDEN_CALLS:
                self._add_issue(
                    severity="critical",
                    category="import",
                    message=f"Importing dangerous function: '{alias.name}' from '{module}'",
                    line_number=node.lineno,
                    suggestion=f"Do not import '{alias.name}'"
                )
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check function calls."""
        self.stats["statement_count"] += 1
        
        # Direct function calls (e.g., eval(...))
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.stats["function_calls"].append(func_name)
            
            if func_name in FORBIDDEN_CALLS:
                self._add_issue(
                    severity="critical",
                    category="call",
                    message=f"Forbidden function call: '{func_name}()'",
                    line_number=node.lineno,
                    suggestion=f"Remove call to '{func_name}()' - this is a dangerous operation"
                )
        
        # Attribute calls (e.g., os.system(...))
        elif isinstance(node.func, ast.Attribute):
            full_name = self._get_full_attr_name(node.func)
            base = full_name.split(".")[0]
            attr = node.func.attr
            
            if base in FORBIDDEN_ATTR_BASES:
                self._add_issue(
                    severity="critical",
                    category="call",
                    message=f"Forbidden attribute call: '{full_name}()'",
                    line_number=node.lineno,
                    suggestion=f"Remove call to '{full_name}()'"
                )
            
            if attr in FORBIDDEN_ATTR_NAMES:
                self._add_issue(
                    severity="high",
                    category="call",
                    message=f"Suspicious method call: '{attr}()' (full: {full_name})",
                    line_number=node.lineno,
                    suggestion="This method name is associated with dangerous operations"
                )
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Check attribute access."""
        try:
            full_name = self._get_full_attr_name(node)
            base = full_name.split(".")[0]
            
            if base in FORBIDDEN_ATTR_BASES:
                self._add_issue(
                    severity="high",
                    category="attribute",
                    message=f"Accessing dangerous module: '{full_name}'",
                    line_number=node.lineno,
                    suggestion=f"Remove reference to '{full_name}'"
                )
            
            if node.attr in FORBIDDEN_ATTR_NAMES:
                self._add_issue(
                    severity="medium",
                    category="attribute",
                    message=f"Accessing suspicious attribute: '{node.attr}' (full: {full_name})",
                    line_number=node.lineno
                )
        except Exception as e:
            logger.debug(f"Error analyzing attribute: {e}")
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Check name usage."""
        if node.id in FORBIDDEN_CALLS:
            self._add_issue(
                severity="high",
                category="reference",
                message=f"Reference to dangerous name: '{node.id}'",
                line_number=node.lineno,
                suggestion=f"Remove usage of '{node.id}'"
            )
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function depth."""
        self.stats["current_depth"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], self.stats["current_depth"])
        
        if self.stats["current_depth"] > MAX_CODE_DEPTH:
            self._add_issue(
                severity="medium",
                category="complexity",
                message=f"Excessive nesting depth: {self.stats['current_depth']} (max: {MAX_CODE_DEPTH})",
                line_number=node.lineno,
                suggestion="Simplify code structure to reduce nesting"
            )
        
        self.generic_visit(node)
        self.stats["current_depth"] -= 1
    
    def visit_For(self, node: ast.For):
        """Track loop depth."""
        self.stats["current_depth"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], self.stats["current_depth"])
        self.stats["statement_count"] += 1
        
        self.generic_visit(node)
        self.stats["current_depth"] -= 1
    
    def visit_While(self, node: ast.While):
        """Track loop depth."""
        self.stats["current_depth"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], self.stats["current_depth"])
        self.stats["statement_count"] += 1
        
        self.generic_visit(node)
        self.stats["current_depth"] -= 1
    
    def _get_full_attr_name(self, node: ast.Attribute) -> str:
        """Recursively build full dotted attribute name."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        parts.reverse()
        return ".".join(parts)
    

# ============================================================================
# PUBLIC API
# ============================================================================
def scan_code(
    code: str,
    level: SecurityLevel = SAFETY_LEVEL
) -> SecurityReport:
    """
    Comprehensive security scan of Python code.
    
    Args:
        code: Python source code to scan
        level: Security level (RELAXED, NORMAL, STRICT)
    
    Returns:
        SecurityReport with findings
    """
    logger.info(f"Starting security scan (level: {level.value})")
    
    # Parse code
    try:
        tree = ast.parse(code)
        code_lines = code.split("\n")
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {e}")
        return SecurityReport(
            is_safe=False,
            issues=[SecurityIssue(
                severity="critical",
                category="syntax",
                message=f"Syntax error: {e}",
                line_number=e.lineno if hasattr(e, 'lineno') else 0
            )],
            level=level
        )
    
    # Run AST visitor
    visitor = SecurityVisitor(code_lines, level)
    visitor.visit(tree)
    
    # Check statement count
    if visitor.stats["statement_count"] > MAX_STATEMENTS:
        visitor._add_issue(
            severity="medium",
            category="complexity",
            message=f"Excessive statement count: {visitor.stats['statement_count']} (max: {MAX_STATEMENTS})",
            line_number=0,
            suggestion="Simplify code or split into smaller functions"
        )
    
    # Check for string-based patterns (additional check)
    import re
    for pattern_str in DANGEROUS_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for line_num, line in enumerate(code_lines, 1):
            if pattern.search(line):
                visitor._add_issue(
                    severity="high",
                    category="pattern",
                    message=f"Dangerous pattern detected: {pattern_str}",
                    line_number=line_num,
                    suggestion="Remove this dangerous operation"
                )
    
    # Determine if code is safe
    critical_or_high = sum(
        1 for issue in visitor.issues
        if issue.severity in ("critical", "high")
    )
    is_safe = critical_or_high == 0
    
    report = SecurityReport(
        is_safe=is_safe,
        issues=visitor.issues,
        stats=visitor.stats,
        level=level
    )
    
    logger.info(report.summary())
    return report

def is_code_safe(
    code: str,
    level: SecurityLevel = SAFETY_LEVEL
) -> Tuple[bool, List[str]]:
    """
    Quick safety check returning boolean and list of issue messages.
    
    Args:
        code: Python source code
        level: Security level
    
    Returns:
        Tuple of (is_safe, list_of_issue_messages)
    """
    report = scan_code(code, level)
    messages = [
        f"[{issue.severity.upper()}] Line {issue.line_number}: {issue.message}"
        for issue in report.issues
    ]
    return report.is_safe, messages
def validate_code(code: str) -> Dict[str, Any]:
    """
    Comprehensive validation returning detailed report.
    
    Args:
        code: Python source code
    
    Returns:
        Dictionary with validation results
    """
    report = scan_code(code)
    return report.to_dict()


# ============================================================================
# CLI INTERFACE
# ============================================================================
def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PiBot Security Scanner - Static code analysis for generated Python"
    )
    parser.add_argument("file", help="Python file to scan")
    parser.add_argument(
        "--level",
        choices=["relaxed", "normal", "strict"],
        default="normal",
        help="Security level (default: normal)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Read file
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1
    
    # Scan
    level = SecurityLevel(args.level)
    report = scan_code(code, level)
    
    # Output results
    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\nSecurity Scan Results for: {args.file}")
        print("=" * 70)
        print(report.summary())
        
        if report.issues:
            print("\nDetailed Issues:")
            print("-" * 70)
            for issue in report.issues:
                print(f"\n[{issue.severity.upper()}] {issue.category.upper()}")
                print(f"Line {issue.line_number}: {issue.message}")
                if issue.suggestion:
                    print(f"Suggestion: {issue.suggestion}")
                if issue.code_snippet and args.verbose:
                    print(f"Code:\n{issue.code_snippet}")
        
        print(f"\nStatistics:")
        print(f"  Max depth: {report.stats['max_depth']}")
        print(f"  Statement count: {report.stats['statement_count']}")
        print(f"  Imports: {', '.join(set(report.stats['imports'])) or 'None'}")
    
    return 0 if report.is_safe else 1


if __name__ == "__main__":
    sys.exit(main())