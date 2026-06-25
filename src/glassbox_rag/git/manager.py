"""
Git Manager — wraps subprocess git for pipeline version control.

Provides detection, auto-commit, status, diff, and PR creation.
Never stores credentials. Read-only detection is always safe.

Usage:
    from glassbox_rag.git import GitManager

    gm = GitManager.from_workspace()
    if gm.is_git_repo():
        print(gm.get_branch())
        gm.commit("Update pipeline config")
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Files GlassBox manages — staged in auto-commit
MANAGED_PATTERNS = [
    ".glassbox/",
    "glassbox.yaml",
    "config/",
]

# Never committed
GITIGNORE_ENTRIES = [
    ".glassbox/cache/",
    ".glassbox/__pycache__/",
]


class GitManager:
    """
    Git integration wrapper using subprocess.

    Wraps git CLI — never stores credentials.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = workspace_path

    @classmethod
    def from_workspace(cls, path: str = ".") -> "GitManager":
        return cls(Path(path).resolve())

    # ── Detection ────────────────────────────────────────────────

    def is_git_repo(self) -> bool:
        """Check if workspace is inside a git repository."""
        return self._run(["git", "rev-parse", "--is-inside-work-tree"]).success

    def get_commit_hash(self) -> str | None:
        """Get current HEAD commit hash."""
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout[:12] if result.success else None

    def get_branch(self) -> str | None:
        """Get current branch name."""
        result = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout if result.success else None

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> list[str]:
        """Get list of changed GlassBox-managed files."""
        result = self._run(["git", "status", "--porcelain"])
        if not result.success:
            return []

        managed = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Line format: "XY filename"
            filepath = line[3:] if len(line) > 3 else ""
            if any(filepath.startswith(p) or filepath == p.rstrip("/")
                   for p in MANAGED_PATTERNS):
                managed.append(line)
        return managed

    def diff(self) -> str:
        """Get human-readable diff of GlassBox config changes."""
        result = self._run(["git", "diff", "--", "glassbox.yaml", ".glassbox/"])
        return result.stdout if result.success else ""

    # ── Commit ───────────────────────────────────────────────────

    def commit(self, message: str = "glassbox: update pipeline state") -> bool:
        """Stage GlassBox-managed files and commit."""
        if not self.is_git_repo():
            logger.warning("Not a git repo — cannot commit")
            return False

        # Stage managed files
        for pattern in MANAGED_PATTERNS:
            full_path = self.workspace / pattern.rstrip("/")
            if full_path.exists():
                self._run(["git", "add", str(full_path)])

        # Check if anything staged
        result = self._run(["git", "diff", "--cached", "--quiet"])
        if result.success:
            logger.info("Nothing to commit")
            return True

        # Commit
        result = self._run(["git", "commit", "-m", message])
        if result.success:
            logger.info("Committed: %s", message)
        else:
            logger.warning("Commit failed: %s", result.stderr)
        return result.success

    # ── Init ─────────────────────────────────────────────────────

    def init_repo(self) -> bool:
        """Initialize a new git repository."""
        result = self._run(["git", "init"])
        if result.success:
            self.setup_gitignore()
        return result.success

    def setup_gitignore(self) -> None:
        """Add GlassBox entries to .gitignore."""
        gitignore_path = self.workspace / ".gitignore"

        existing = ""
        if gitignore_path.exists():
            existing = gitignore_path.read_text(encoding="utf-8")

        additions = []
        for entry in GITIGNORE_ENTRIES:
            if entry not in existing:
                additions.append(entry)

        if additions:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n# GlassBox RAG\n")
                for entry in additions:
                    f.write(f"{entry}\n")
            logger.info("Updated .gitignore with GlassBox entries")

    # ── PR ───────────────────────────────────────────────────────

    def open_pr(self, title: str = "GlassBox pipeline update", body: str = "") -> bool:
        """Create a PR using gh CLI. Silent skip if gh not available."""
        result = self._run(["gh", "pr", "create", "--title", title, "--body", body or title])
        if result.success:
            logger.info("PR created: %s", title)
        else:
            logger.debug("PR creation skipped (gh not available or failed)")
        return result.success

    # ── Internal ─────────────────────────────────────────────────

    @dataclass
    class _RunResult:
        success: bool
        stdout: str
        stderr: str

    def _run(self, cmd: list[str], timeout: int = 15) -> _RunResult:
        """Run a subprocess command and return result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
                timeout=timeout,
            )
            return self._RunResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        except FileNotFoundError:
            return self._RunResult(success=False, stdout="", stderr="command not found")
        except subprocess.TimeoutExpired:
            return self._RunResult(success=False, stdout="", stderr="timeout")
