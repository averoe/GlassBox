"""
Golden Datasets — YAML-defined query/answer/document triplets for evaluation.

Stored in `.glassbox/datasets/`. Versioned alongside pipeline config.

Format:
    - query: "What is the refund policy?"
      expected_answer: "Refunds are processed within 5 business days."
      relevant_doc_ids: ["doc_001", "doc_002"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List

import yaml

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GoldenEntry:
    """A single query/answer/document triplet."""

    query: str
    expected_answer: str = ""
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class GoldenDataset:
    """
    YAML-defined golden dataset for evaluation.

    Provides iteration and serialization for query/answer/doc_id triplets.
    """

    def __init__(self, entries: list[GoldenEntry] | None = None) -> None:
        self.entries: list[GoldenEntry] = entries or []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoldenDataset":
        """Load a golden dataset from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError(f"Dataset must be a YAML list, got {type(data).__name__}")

        entries = []
        for item in data:
            entries.append(GoldenEntry(
                query=item.get("query", ""),
                expected_answer=item.get("expected_answer", ""),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                metadata=item.get("metadata", {}),
            ))

        logger.info("Loaded golden dataset from %s: %d entries", path, len(entries))
        return cls(entries)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the dataset to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for entry in self.entries:
            item: dict[str, Any] = {"query": entry.query}
            if entry.expected_answer:
                item["expected_answer"] = entry.expected_answer
            if entry.relevant_doc_ids:
                item["relevant_doc_ids"] = entry.relevant_doc_ids
            if entry.metadata:
                item["metadata"] = entry.metadata
            data.append(item)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("Saved golden dataset to %s: %d entries", path, len(self.entries))

    def __iter__(self) -> Iterator[GoldenEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> GoldenEntry:
        return self.entries[index]

    def add(self, query: str, expected_answer: str = "", relevant_doc_ids: list[str] | None = None) -> None:
        """Add an entry to the dataset."""
        self.entries.append(GoldenEntry(
            query=query,
            expected_answer=expected_answer,
            relevant_doc_ids=relevant_doc_ids or [],
        ))
