"""Syntax-check generated CARLA Python scripts.

Stage 2b of the pure-LLM risk pipeline turns the validated DSL into an executable
CARLA Python script deterministically, so the script is expected to be
syntactically valid by construction. This module provides a cheap sanity gate
that compiles the produced file to byte code with the standard-library
``py_compile``.

``py_compile`` only parses + compiles to byte code; it does NOT execute the
module body, so the missing ``carla`` runtime dependency does not cause a false
failure. The return structure mirrors ``tools/scenic_compile.check_scenic_compile``.
"""

import py_compile
from typing import Any, Dict


def check_python_compile(py_path: str) -> Dict[str, Any]:
    """Return ``{compiled, error}`` for one generated ``.py`` file."""
    try:
        py_compile.compile(py_path, doraise=True)
    except py_compile.PyCompileError as exc:
        return {"compiled": False, "error": _trim(str(exc))}
    except (FileNotFoundError, OSError) as exc:
        return {"compiled": False, "error": _trim(str(exc))}
    return {"compiled": True, "error": None}


def _trim(text: str, limit: int = 4000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"...{text[-limit:]}"
