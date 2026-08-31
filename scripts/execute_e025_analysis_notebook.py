"""Execute the E025 analysis notebook with only the Python standard library.

The experiment environment deliberately does not install Jupyter.  This small,
task-specific runner executes Python code cells in order, captures their text
outputs, and writes a valid executed notebook for the retained audit trail.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO_ROOT
    / "artifacts"
    / "runs"
    / "experiment-025-20260819T013016Z"
    / "preflight"
    / "attempt-006-acquisition-analysis.ipynb"
)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
    }
    execution_count = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        outputs: list[dict[str, object]] = []
        try:
            tree = ast.parse(source, filename=str(NOTEBOOK), mode="exec")
            expression = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expression = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(expression)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stdout
            ):
                exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
                value = (
                    eval(compile(expression, str(NOTEBOOK), "eval"), namespace)
                    if expression is not None
                    else None
                )
            if stdout.getvalue():
                outputs.append(
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": stdout.getvalue(),
                    }
                )
            if expression is not None and value is not None:
                outputs.append(
                    {
                        "data": {"text/plain": repr(value)},
                        "execution_count": execution_count,
                        "metadata": {},
                        "output_type": "execute_result",
                    }
                )
        except BaseException as exc:
            outputs.append(
                {
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            NOTEBOOK.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    notebook.setdefault("metadata", {})["execution"] = {
        "runner": "scripts/execute_e025_analysis_notebook.py",
        "status": "PASS",
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"executed {execution_count} code cells: {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
