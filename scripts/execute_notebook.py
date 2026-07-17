"""Build and execute the project notebook in one process.

The notebook uses ordinary Python cells only. An in-process runner avoids a
Jupyter transport dependency while preserving deterministic top-to-bottom
execution, captured streams, rich DataFrame output and embedded PNG output.
"""

from __future__ import annotations

import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import nbformat

from scripts.build_notebook import OUTPUT, ROOT, build


def rich_data(value: Any) -> dict[str, str]:
    data: dict[str, str] = {"text/plain": repr(value)}
    if hasattr(value, "_repr_html_"):
        html = value._repr_html_()
        if html:
            data["text/html"] = html
    if hasattr(value, "_repr_png_"):
        png = value._repr_png_()
        if png:
            data["image/png"] = (
                png if isinstance(png, str) else base64.b64encode(png).decode("ascii")
            )
    return data


def execute_cell(source: str, namespace: dict[str, Any], outputs: list[Any]) -> Any:
    tree = ast.parse(source, mode="exec")
    final_expression = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final_expression = ast.Expression(tree.body.pop().value)
    if tree.body:
        module = ast.Module(body=tree.body, type_ignores=[])
        exec(compile(module, "<notebook-cell>", "exec"), namespace)
    if final_expression is not None:
        return eval(compile(final_expression, "<notebook-cell>", "eval"), namespace)
    return None


def execute() -> None:
    build()
    notebook = nbformat.read(OUTPUT, as_version=4)
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(OUTPUT)}
    execution_count = 0
    previous_cwd = Path.cwd()

    def display(*values: Any) -> None:
        active_outputs.extend(
            nbformat.v4.new_output("display_data", data=rich_data(value), metadata={})
            for value in values
        )

    try:
        import os

        os.chdir(ROOT)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            active_outputs = cell.outputs
            namespace["display"] = display
            stdout, stderr = StringIO(), StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    value = execute_cell(cell.source, namespace, active_outputs)
            except Exception as exc:
                active_outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(exc).__name__,
                        evalue=str(exc),
                        traceback=[],
                    )
                )
                nbformat.write(notebook, OUTPUT)
                raise
            if stdout.getvalue():
                active_outputs.insert(
                    0, nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue())
                )
            if stderr.getvalue():
                active_outputs.insert(
                    0, nbformat.v4.new_output("stream", name="stderr", text=stderr.getvalue())
                )
            if value is not None:
                active_outputs.append(
                    nbformat.v4.new_output(
                        "execute_result",
                        data=rich_data(value),
                        metadata={},
                        execution_count=execution_count,
                    )
                )
    finally:
        import os

        os.chdir(previous_cwd)

    notebook.metadata["execution"] = {
        "status": "completed",
        "runner": "in-process nbformat runner",
    }
    nbformat.write(notebook, OUTPUT)
    print(f"Executed notebook: {OUTPUT}")


if __name__ == "__main__":
    execute()
