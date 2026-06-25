"""
Version Manager — pipeline versioning, prompt versioning, and snapshots.

All version data lives in `.glassbox/` — plain YAML and JSON,
human-readable, git-committable. No database required.

Usage:
    from glassbox_rag import VersionManager

    vm = VersionManager.from_workspace()
    vm.snapshot(tag="v1-baseline")
    diff = vm.diff("v1-baseline", "v2-experiment")
    vm.rollback("v1-baseline")
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineVersion:
    """A recorded pipeline config version."""

    version: int
    config_hash: str
    timestamp: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptVersion:
    """A recorded prompt version."""

    name: str
    version: int
    content_hash: str
    timestamp: str
    content: str = ""


@dataclass
class Snapshot:
    """A frozen point-in-time snapshot of the full pipeline state."""

    tag: str
    timestamp: str
    pipeline_version: int
    config_hash: str
    prompt_versions: dict[str, int] = field(default_factory=dict)
    dataset_versions: dict[str, str] = field(default_factory=dict)
    evaluation_scores: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "config_hash": self.config_hash,
            "prompt_versions": self.prompt_versions,
            "dataset_versions": self.dataset_versions,
            "evaluation_scores": self.evaluation_scores,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Snapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class VersionManager:
    """
    Manages pipeline versioning, prompt versioning, and snapshots.

    All state is stored in `.glassbox/` as human-readable YAML/JSON files.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = workspace_path
        self.glassbox_dir = workspace_path / ".glassbox"
        self.versions_dir = self.glassbox_dir / "versions"
        self.prompts_dir = self.glassbox_dir / "prompts"
        self.snapshots_dir = self.glassbox_dir / "snapshots"
        self.datasets_dir = self.glassbox_dir / "datasets"

        # Ensure dirs exist
        for d in [self.versions_dir, self.prompts_dir, self.snapshots_dir, self.datasets_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(cls, path: str = ".") -> "VersionManager":
        """Create a VersionManager for the given workspace directory."""
        return cls(Path(path).resolve())

    # ── Pipeline Versioning ──────────────────────────────────────

    def _config_path(self) -> Path | None:
        """Find the pipeline config file."""
        candidates = [
            self.workspace / "glassbox.yaml",
            self.workspace / "config" / "default.yaml",
            self.workspace / "config" / "glassbox.yaml",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _hash_file(self, path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def get_current_pipeline_version(self) -> int:
        """Get the current pipeline version number."""
        version_file = self.versions_dir / "current.json"
        if version_file.exists():
            data = json.loads(version_file.read_text(encoding="utf-8"))
            return data.get("version", 0)
        return 0

    def _set_pipeline_version(self, version: int, config_hash: str, config_data: dict) -> None:
        """Set the current pipeline version."""
        now = datetime.now(timezone.utc).isoformat()

        # Update current pointer
        current_file = self.versions_dir / "current.json"
        current_file.write_text(
            json.dumps({"version": version, "config_hash": config_hash, "timestamp": now}, indent=2),
            encoding="utf-8",
        )

        # Archive this version
        archive_file = self.versions_dir / f"v{version}.yaml"
        archive_data = {
            "version": version,
            "config_hash": config_hash,
            "timestamp": now,
            "config": config_data,
        }
        archive_file.write_text(yaml.dump(archive_data, default_flow_style=False), encoding="utf-8")

    def check_and_increment_version(self) -> int:
        """
        Check if config has changed and auto-increment version if so.

        Returns the current version number.
        """
        config_path = self._config_path()
        if config_path is None:
            return 0

        current_hash = self._hash_file(config_path)
        current_version = self.get_current_pipeline_version()

        # Check if hash matches current
        current_file = self.versions_dir / "current.json"
        if current_file.exists():
            data = json.loads(current_file.read_text(encoding="utf-8"))
            if data.get("config_hash") == current_hash:
                return current_version

        # Config changed — increment
        new_version = current_version + 1
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self._set_pipeline_version(new_version, current_hash, config_data)

        logger.info("Pipeline version incremented: v%d → v%d", current_version, new_version)
        return new_version

    def list_versions(self) -> list[PipelineVersion]:
        """List all stored pipeline versions."""
        versions = []
        for f in sorted(self.versions_dir.glob("v*.yaml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            versions.append(PipelineVersion(
                version=data.get("version", 0),
                config_hash=data.get("config_hash", ""),
                timestamp=data.get("timestamp", ""),
                config_snapshot=data.get("config", {}),
            ))
        return versions

    # ── Prompt Versioning ────────────────────────────────────────

    def get_prompt_versions(self) -> dict[str, PromptVersion]:
        """Get current versions of all tracked prompts."""
        versions: dict[str, PromptVersion] = {}
        index_file = self.prompts_dir / "index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            for name, info in data.items():
                versions[name] = PromptVersion(
                    name=name,
                    version=info.get("version", 1),
                    content_hash=info.get("content_hash", ""),
                    timestamp=info.get("timestamp", ""),
                )
        return versions

    def track_prompt(self, name: str, content: str) -> int:
        """
        Track a prompt version. If content changed, increment version.

        Returns the current version number for this prompt.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        index_file = self.prompts_dir / "index.json"

        index: dict[str, Any] = {}
        if index_file.exists():
            index = json.loads(index_file.read_text(encoding="utf-8"))

        current = index.get(name, {})
        if current.get("content_hash") == content_hash:
            return current.get("version", 1)

        new_version = current.get("version", 0) + 1
        now = datetime.now(timezone.utc).isoformat()

        index[name] = {
            "version": new_version,
            "content_hash": content_hash,
            "timestamp": now,
        }
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

        # Archive prompt content
        prompt_file = self.prompts_dir / f"{name}_v{new_version}.txt"
        prompt_file.write_text(content, encoding="utf-8")

        logger.info("Prompt '%s' versioned: v%d", name, new_version)
        return new_version

    # ── Snapshots ────────────────────────────────────────────────

    def snapshot(self, tag: str) -> Snapshot:
        """
        Freeze the full pipeline state at a point in time.

        Stores config version, prompt versions, dataset versions,
        evaluation scores, and git commit hash.
        """
        pipeline_version = self.check_and_increment_version()
        config_path = self._config_path()
        config_hash = self._hash_file(config_path) if config_path else ""

        # Prompt versions
        prompt_versions = {
            name: pv.version
            for name, pv in self.get_prompt_versions().items()
        }

        # Dataset versions (hash each file in datasets/)
        dataset_versions: dict[str, str] = {}
        for ds_file in self.datasets_dir.glob("*.yaml"):
            dataset_versions[ds_file.name] = self._hash_file(ds_file)

        # Evaluation scores (latest)
        eval_scores = self._get_latest_eval_scores()

        # Git commit hash
        git_commit = self._get_git_commit()

        snap = Snapshot(
            tag=tag,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_version=pipeline_version,
            config_hash=config_hash,
            prompt_versions=prompt_versions,
            dataset_versions=dataset_versions,
            evaluation_scores=eval_scores,
            git_commit=git_commit,
        )

        # Write snapshot
        snap_file = self.snapshots_dir / f"{tag}.yaml"
        snap_file.write_text(
            yaml.dump(snap.to_dict(), default_flow_style=False),
            encoding="utf-8",
        )

        logger.info("Snapshot created: %s (pipeline v%d)", tag, pipeline_version)
        return snap

    def get_snapshot(self, tag: str) -> Snapshot | None:
        """Load a snapshot by tag."""
        snap_file = self.snapshots_dir / f"{tag}.yaml"
        if not snap_file.exists():
            return None
        data = yaml.safe_load(snap_file.read_text(encoding="utf-8")) or {}
        return Snapshot.from_dict(data)

    def list_snapshots(self) -> list[Snapshot]:
        """List all stored snapshots."""
        snaps = []
        for f in sorted(self.snapshots_dir.glob("*.yaml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            snaps.append(Snapshot.from_dict(data))
        return snaps

    def diff(self, tag_a: str, tag_b: str) -> dict[str, Any]:
        """
        Compare two snapshots and return differences.

        Returns a dict with changed fields and their before/after values.
        """
        snap_a = self.get_snapshot(tag_a)
        snap_b = self.get_snapshot(tag_b)
        if snap_a is None:
            raise ValueError(f"Snapshot '{tag_a}' not found")
        if snap_b is None:
            raise ValueError(f"Snapshot '{tag_b}' not found")

        dict_a = snap_a.to_dict()
        dict_b = snap_b.to_dict()

        changes: dict[str, Any] = {}
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        for key in all_keys:
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)
            if val_a != val_b:
                changes[key] = {"before": val_a, "after": val_b}

        # Config-level diff if versions are available
        if snap_a.pipeline_version != snap_b.pipeline_version:
            ver_a = self._load_version_config(snap_a.pipeline_version)
            ver_b = self._load_version_config(snap_b.pipeline_version)
            if ver_a and ver_b:
                config_diff = self._dict_diff(ver_a, ver_b)
                if config_diff:
                    changes["config_changes"] = config_diff

        return {
            "snapshot_a": tag_a,
            "snapshot_b": tag_b,
            "changes": changes,
        }

    def rollback(self, tag: str) -> None:
        """
        Restore pipeline config from a snapshot.

        Overwrites the current config file with the config stored
        in the snapshot's corresponding pipeline version.
        """
        snap = self.get_snapshot(tag)
        if snap is None:
            raise ValueError(f"Snapshot '{tag}' not found")

        config_data = self._load_version_config(snap.pipeline_version)
        if config_data is None:
            raise ValueError(
                f"Config for pipeline version {snap.pipeline_version} not found"
            )

        config_path = self._config_path()
        if config_path is None:
            config_path = self.workspace / "glassbox.yaml"

        config_path.write_text(
            yaml.dump(config_data, default_flow_style=False),
            encoding="utf-8",
        )

        logger.info("Rolled back to snapshot '%s' (pipeline v%d)", tag, snap.pipeline_version)

    # ── Internal helpers ─────────────────────────────────────────

    def _load_version_config(self, version: int) -> dict[str, Any] | None:
        """Load the config from a stored pipeline version."""
        version_file = self.versions_dir / f"v{version}.yaml"
        if not version_file.exists():
            return None
        data = yaml.safe_load(version_file.read_text(encoding="utf-8")) or {}
        return data.get("config", {})

    def _get_latest_eval_scores(self) -> dict[str, Any]:
        """Get the latest evaluation scores from .glassbox/evaluations/."""
        eval_dir = self.glassbox_dir / "evaluations"
        if not eval_dir.exists():
            return {}
        files = sorted(eval_dir.glob("*.json"), reverse=True)
        if not files:
            return {}
        try:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            return data.get("aggregate_scores", {})
        except Exception:
            return {}

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash, if in a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _dict_diff(
        self, a: dict[str, Any], b: dict[str, Any], prefix: str = ""
    ) -> dict[str, Any]:
        """Recursive diff of two dicts."""
        changes: dict[str, Any] = {}
        all_keys = set(a.keys()) | set(b.keys())
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            val_a = a.get(key)
            val_b = b.get(key)
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                nested = self._dict_diff(val_a, val_b, full_key)
                changes.update(nested)
            elif val_a != val_b:
                changes[full_key] = {"before": val_a, "after": val_b}
        return changes
