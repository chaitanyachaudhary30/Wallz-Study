"""Single Source of Truth Architectural Guard (P05).

Verifies that no component outside WallzEngine (including Greedy agent and web backends)
re-implements or bypasses game rules, pathfinding, or mutates engine state attributes.
"""
from __future__ import annotations

import ast
import os
import re
from typing import List


FORBIDDEN_ATTRS = {
    "pawns",
    "walls",
    "blocked_cells",
    "wall_inventory",
    "hop_available",
    "block_available",
    "latest_wall",
    "current_player",
    "winner",
    "is_draw",
}


class SingleSourceASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[str] = []

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._check_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._check_target(node.target, node.lineno)
        self.generic_visit(node)

    def _check_target(self, target: ast.AST, lineno: int):
        if isinstance(target, ast.Attribute):
            if target.attr in FORBIDDEN_ATTRS:
                self.violations.append(
                    f"{self.filename}:{lineno}: Forbidden direct mutation of engine state attribute '{target.attr}'"
                )

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if "pathfinding" in alias.name:
                self.violations.append(
                    f"{self.filename}:{node.lineno}: Forbidden import of pathfinding module '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and "pathfinding" in node.module:
            self.violations.append(
                f"{self.filename}:{node.lineno}: Forbidden import from pathfinding module '{node.module}'"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id == "deque" and isinstance(node.ctx, ast.Load):
            self.violations.append(
                f"{self.filename}:{node.lineno}: Forbidden use of 'deque' (BFS re-implementation smell)"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr == "deque" and isinstance(node.ctx, ast.Load):
            self.violations.append(
                f"{self.filename}:{node.lineno}: Forbidden use of 'collections.deque' (BFS re-implementation smell)"
            )
        self.generic_visit(node)


def test_single_source_python_guard():
    """Scans agents/greedy.py and web/backend/**/*.py for rule re-implementation or state mutation."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_files = [os.path.join(repo_root, "agents", "greedy.py")]

    web_backend_dir = os.path.join(repo_root, "web", "backend")
    if os.path.exists(web_backend_dir):
        for root, _, files in os.walk(web_backend_dir):
            for f in files:
                if f.endswith(".py"):
                    python_files.append(os.path.join(root, f))

    all_violations = []
    for fpath in python_files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), fpath)
        visitor = SingleSourceASTVisitor(fpath)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, "Single-source architectural violations detected:\n" + "\n".join(all_violations)


def test_single_source_catches_mutation():
    """Proves the AST guard correctly detects illegal mutation of engine state."""
    code = "engine.pawns = (1, 2)"
    tree = ast.parse(code, "<test_dummy>")
    visitor = SingleSourceASTVisitor("<test_dummy>")
    visitor.visit(tree)
    assert len(visitor.violations) == 1
    assert "Forbidden direct mutation of engine state attribute 'pawns'" in visitor.violations[0]


def test_single_source_frontend_guard():
    """Scans web/frontend/src/**/*.ts(x) for rule logic duplication if frontend exists."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    frontend_src = os.path.join(repo_root, "web", "frontend", "src")
    if not os.path.exists(frontend_src):
        return  # Skipped until frontend is created in P11

    pattern = re.compile(r"\bbfs\b|shortestPath\(|isLegal|canPlace|wallEdges.*block")
    violations = []
    for root, _, files in os.walk(frontend_src):
        for f in files:
            if f.endswith((".ts", ".tsx")):
                fpath = os.path.join(root, f)
                with open(fpath, "r", encoding="utf-8") as file_obj:
                    for line_no, line in enumerate(file_obj, start=1):
                        if pattern.search(line):
                            violations.append(f"{fpath}:{line_no}: {line.strip()}")

    assert not violations, "Frontend rule duplication violations detected:\n" + "\n".join(violations)
