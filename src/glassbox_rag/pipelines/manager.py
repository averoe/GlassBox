"""
Pipeline Manager — multi-pipeline workspace management.

Detects, lists, switches, and compares pipeline configs within a workspace.
Workspace state is stored in `.glassbox/workspace.yaml`.

Usage:
    from glassbox_rag.pipelines import PipelineManager

    pm = PipelineManager.from_workspace()
    pipelines = pm.detect()  # scan for pipeline configs
    pm.switch("experiment-a")
    active = pm.get_active()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Config file names we look for during detection
CONFIG_CANDIDATES = [
    "glassbox.yaml",
    "glassbox.yml",
    "config/default.yaml",
    "config/glassbox.yaml",
]


@dataclass
class PipelineInfo:
    """Information about a registered pipeline."""

    name: str
    config_path: str
    status: str = "inactive"  # active / inactive / unknown
    last_run: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineManager:
    """
    Manages multiple pipeline configurations within a workspace.

    Each pipeline is a separate YAML config. The active pipeline is
    tracked in `.glassbox/workspace.yaml`.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = workspace_path
        self.glassbox_dir = workspace_path / ".glassbox"
        self.workspace_file = self.glassbox_dir / "workspace.yaml"
        self.glassbox_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(cls, path: str = ".") -> "PipelineManager":
        return cls(Path(path).resolve())

    # ── Detection ────────────────────────────────────────────────

    def detect(self, directory: str | None = None) -> list[PipelineInfo]:
        """
        Scan a directory for valid pipeline configs.

        Finds files named glassbox.yaml, glassbox.yml, or configs
        in config/ subdirectory. Returns PipelineInfo for each found.
        """
        scan_dir = Path(directory) if directory else self.workspace
        found: list[PipelineInfo] = []

        # Check standard locations
        for candidate in CONFIG_CANDIDATES:
            path = scan_dir / candidate
            if path.exists():
                name = path.stem if path.stem != "default" else path.parent.name
                found.append(PipelineInfo(
                    name=name,
                    config_path=str(path.relative_to(scan_dir)),
                ))

        # Check config/ directory for multiple configs
        config_dir = scan_dir / "config"
        if config_dir.is_dir():
            for f in config_dir.glob("*.yaml"):
                if f.name not in ("default.yaml", "glassbox.yaml"):
                    rel = str(f.relative_to(scan_dir))
                    if not any(p.config_path == rel for p in found):
                        found.append(PipelineInfo(
                            name=f.stem,
                            config_path=rel,
                        ))
            for f in config_dir.glob("*.yml"):
                rel = str(f.relative_to(scan_dir))
                if not any(p.config_path == rel for p in found):
                    found.append(PipelineInfo(
                        name=f.stem,
                        config_path=rel,
                    ))

        # Mark active pipeline
        active = self._get_active_name()
        for p in found:
            if p.name == active:
                p.status = "active"

        logger.info("Detected %d pipeline(s) in %s", len(found), scan_dir)
        return found

    # ── List / Get ───────────────────────────────────────────────

    def list_registered(self) -> list[PipelineInfo]:
        """List pipelines registered in workspace.yaml."""
        data = self._load_workspace()
        pipelines = data.get("pipelines", [])
        active = data.get("active_pipeline", "default")

        result = []
        for p in pipelines:
            info = PipelineInfo(
                name=p.get("name", "unknown"),
                config_path=p.get("config_path", ""),
                status="active" if p.get("name") == active else "inactive",
                last_run=p.get("last_run"),
            )
            result.append(info)
        return result

    def get_active(self) -> PipelineInfo | None:
        """Get the currently active pipeline."""
        pipelines = self.list_registered()
        for p in pipelines:
            if p.status == "active":
                return p
        return None

    # ── Switch ───────────────────────────────────────────────────

    def switch(self, name: str) -> bool:
        """Switch the active pipeline."""
        data = self._load_workspace()
        pipelines = data.get("pipelines", [])

        found = any(p.get("name") == name for p in pipelines)
        if not found:
            # Auto-register if we can detect it
            detected = self.detect()
            match = next((d for d in detected if d.name == name), None)
            if match:
                pipelines.append({
                    "name": match.name,
                    "config_path": match.config_path,
                })
                logger.info("Auto-registered pipeline '%s'", name)
            else:
                logger.warning("Pipeline '%s' not found", name)
                return False

        data["active_pipeline"] = name
        data["pipelines"] = pipelines
        self._save_workspace(data)
        logger.info("Switched active pipeline to '%s'", name)
        return True

    # ── Register ─────────────────────────────────────────────────

    def register(self, name: str, config_path: str) -> None:
        """Register a pipeline in the workspace."""
        data = self._load_workspace()
        pipelines = data.get("pipelines", [])

        # Check for duplicate
        if any(p.get("name") == name for p in pipelines):
            logger.info("Pipeline '%s' already registered", name)
            return

        pipelines.append({
            "name": name,
            "config_path": config_path,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        })
        data["pipelines"] = pipelines
        self._save_workspace(data)
        logger.info("Registered pipeline '%s' → %s", name, config_path)

    # ── Compare ──────────────────────────────────────────────────

    def compare(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Compare two pipeline configs."""
        pipelines = {p.get("name"): p for p in self._load_workspace().get("pipelines", [])}
        path_a = pipelines.get(name_a, {}).get("config_path")
        path_b = pipelines.get(name_b, {}).get("config_path")

        if not path_a or not path_b:
            raise ValueError(f"Pipeline not found: {name_a if not path_a else name_b}")

        config_a = yaml.safe_load((self.workspace / path_a).read_text(encoding="utf-8")) or {}
        config_b = yaml.safe_load((self.workspace / path_b).read_text(encoding="utf-8")) or {}

        return _dict_diff(config_a, config_b)

    # ── Internal ─────────────────────────────────────────────────

    def _get_active_name(self) -> str:
        """Get the name of the active pipeline."""
        data = self._load_workspace()
        return data.get("active_pipeline", "default")

    def _load_workspace(self) -> dict[str, Any]:
        """Load workspace.yaml."""
        if not self.workspace_file.exists():
            return {"active_pipeline": "default", "pipelines": []}
        return yaml.safe_load(self.workspace_file.read_text(encoding="utf-8")) or {}

    def _save_workspace(self, data: dict[str, Any]) -> None:
        """Save workspace.yaml."""
        self.workspace_file.write_text(
            yaml.dump(data, default_flow_style=False),
            encoding="utf-8",
        )


def _dict_diff(a: dict, b: dict, prefix: str = "") -> dict[str, Any]:
    """Recursive diff of two config dicts."""
    changes: dict[str, Any] = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        val_a = a.get(key)
        val_b = b.get(key)
        if isinstance(val_a, dict) and isinstance(val_b, dict):
            changes.update(_dict_diff(val_a, val_b, full_key))
        elif val_a != val_b:
            changes[full_key] = {"before": val_a, "after": val_b}
    return changes
