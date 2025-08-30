#!/usr/bin/env python3
"""
Clean a Jupyter notebook by:
 - Removing Python comments from code cells
 - Removing docstrings and standalone string-only statements (e.g., "Notes: ...")
 - Ensuring matplotlib figures are displayed inline by inserting plt.show()
   before plt.close() or after savefig() when needed.

Usage:
  python tools/clean_notebook.py ASSignment.ipynb

This script overwrites the input notebook after creating a backup
with suffix `.bak` in the same directory.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tokenize
import ast
from typing import List


def strip_python_comments(code: str) -> str:
    """Return code with all Python comments removed using tokenize.

    - Removes both full-line comments and trailing inline comments (# ...)
    - Preserves strings and whitespace
    """
    if "#" not in code:
        return code

    out_tokens = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        return code

    for tok in tokens:
        tok_type, tok_str, start, end, line = tok
        if tok_type == tokenize.COMMENT:
            continue
        out_tokens.append(tok)

    try:
        cleaned = tokenize.untokenize(out_tokens)
    except Exception:
        return code

    return cleaned


def remove_docstrings_and_bare_strings(code: str) -> str:
    """Remove docstrings and bare string expression statements using AST.

    - Strips module, class, and function docstrings.
    - Removes any top-level bare string expressions (notes-as-strings).
    """
    try:
        tree = ast.parse(code)

        class StripDocstrings(ast.NodeTransformer):
            def _strip_docstring(self, body: List[ast.stmt]) -> List[ast.stmt]:
                # Remove leading docstring if present
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), (ast.Constant, ast.Str)):
                    val = body[0].value
                    s = val.value if isinstance(val, ast.Constant) else val.s
                    if isinstance(s, str):
                        body = body[1:]
                # Remove any bare string expressions anywhere in the block
                new_body: List[ast.stmt] = []
                for stmt in body:
                    if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, 'value', None), (ast.Constant, ast.Str)):
                        val = stmt.value
                        s = val.value if isinstance(val, ast.Constant) else val.s
                        if isinstance(s, str):
                            continue
                    new_body.append(stmt)
                return new_body

            def visit_Module(self, node: ast.Module):
                self.generic_visit(node)
                node.body = self._strip_docstring(list(node.body))
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.generic_visit(node)
                node.body = self._strip_docstring(list(node.body))
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.generic_visit(node)
                node.body = self._strip_docstring(list(node.body))
                return node

            def visit_ClassDef(self, node: ast.ClassDef):
                self.generic_visit(node)
                node.body = self._strip_docstring(list(node.body))
                return node

        tree = StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return code


def ensure_plots_show(code: str) -> str:
    """Insert plt.show() so figures render inline in notebooks.

    Rules:
    - For any line that contains 'plt.close()', ensure a 'plt.show()' occurs
      just before it (on the same line if semicolon-chained or as a separate line).
    - If the cell contains a 'savefig(' but no 'plt.close()' and no 'plt.show()',
      insert a 'plt.show()' right after the first 'savefig(' line.
    """
    if "plt.close()" not in code and "savefig(" not in code:
        return code

    lines = code.splitlines(keepends=True)

    new_lines: List[str] = []
    savefig_line_idxs: List[int] = []

    for i, ln in enumerate(lines):
        if "savefig(" in ln:
            savefig_line_idxs.append(len(new_lines))

        if "plt.close()" in ln and "plt.show()" not in ln:
            ln = ln.replace("plt.close()", "plt.show(); plt.close()")
            new_lines.append(ln)
            continue

        if ln.strip() == "plt.close()" and "plt.show()" not in ln:
            new_lines.append("plt.show()\n")
            new_lines.append(ln)
            continue

        new_lines.append(ln)

    if savefig_line_idxs and not any("plt.show()" in ln for ln in new_lines):
        idx = savefig_line_idxs[0] + 1
        new_lines.insert(idx, "plt.show()\n")

    return "".join(new_lines)


def process_notebook(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    backup_path = path + ".bak"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        src = cell.get("source", [])
        code = "".join(src) if isinstance(src, list) else str(src)
        orig_code = code

        code = strip_python_comments(code)
        code = remove_docstrings_and_bare_strings(code)
        code = ensure_plots_show(code)

        if code != orig_code:
            changed = True
            cell["source"] = code.splitlines(keepends=True)

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False)
        print(f"Updated notebook saved to {path}. Backup at {backup_path}")
    else:
        print("No changes were necessary.")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python tools/clean_notebook.py <notebook.ipynb>")
        return 2
    path = argv[1]
    if not os.path.isfile(path):
        print(f"Notebook not found: {path}")
        return 1
    process_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

