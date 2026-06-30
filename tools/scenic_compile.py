"""Compile-check generated Scenic programs without polluting the autoscenario env.

Scenic (``scenic==3.0.0b2``) lives in a separate conda env (default ``scenicNL``),
so we shell out to ``conda run -n <env> python`` and call
``scenic.scenarioFromFile(path, mode2D=True)`` -- the same check scenicNL uses to
count an "executable" program. This validates syntax + semantics + map loading
but does NOT sample a concrete scene (so absolute placements that may be off-road
do not fail here).
"""

import subprocess
from typing import Any, Dict, List, Optional


_CHECK_SNIPPET = (
    "import sys, scenic; "
    "scenic.scenarioFromFile(sys.argv[1], mode2D=True); "
    "print('SCENIC_COMPILE_OK')"
)


def build_compile_command(scenic_path: str, conda_env: str) -> List[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-c",
        _CHECK_SNIPPET,
        scenic_path,
    ]


def check_scenic_compile(
    scenic_path: str,
    conda_env: str = "scenicNL",
    timeout: int = 240,
) -> Dict[str, Any]:
    """Return ``{compiled, returncode, error}`` for one ``.scenic`` file."""
    cmd = build_compile_command(scenic_path, conda_env)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {
            "compiled": False,
            "returncode": None,
            "error": (
                "`conda` executable not found; cannot run Scenic compile check. "
                f"Install conda or expose env `{conda_env}`."
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "returncode": None,
            "error": f"Scenic compile check timed out after {timeout}s.",
        }

    compiled = proc.returncode == 0 and "SCENIC_COMPILE_OK" in (proc.stdout or "")
    error: Optional[str] = None
    if not compiled:
        error = _trim((proc.stderr or "").strip() or (proc.stdout or "").strip())
    return {
        "compiled": compiled,
        "returncode": proc.returncode,
        "error": error,
    }


def _trim(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"...{text[-limit:]}"
