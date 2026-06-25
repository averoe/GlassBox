"""
TraceTree widget — hierarchical trace step visualisation using Textual Tree.
"""

from __future__ import annotations

from typing import Any, Dict

from textual.widgets import Tree


class TraceTree(Tree):
    """
    Displays a trace's step hierarchy in a navigable tree.

    Nodes are colour-coded:
      ✓ green  — success
      ❌ red   — error
    """

    DEFAULT_CSS = """
    TraceTree {
        background: #181b27;
        min-height: 10;
        border: round #2a2e42;
        padding: 1;
        scrollbar-background: #12141c;
        scrollbar-color: #2a2e42;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("Trace Steps", **kwargs)
        self.guide_depth = 3

    def load_trace(self, trace_data: Dict[str, Any]) -> None:
        """Populate the tree from a trace dict (from engine.get_trace)."""
        self.clear()
        root_step = trace_data.get("root_step")
        if not root_step:
            self.root.add_leaf("(no steps recorded)")
            return

        trace_id = trace_data.get("trace_id", "")[:12]
        dur = trace_data.get("duration_ms", 0)
        self.root.set_label(f"Trace {trace_id}…  ({dur:.1f}ms)")
        self._add_step(self.root, root_step)
        self.root.expand_all()

    def _add_step(self, parent, step: Dict[str, Any]) -> None:
        name = step.get("name", "?")
        ms = step.get("duration_ms", 0)
        error = step.get("error")

        if error:
            label = f"❌ {name}  ({ms:.1f}ms) — {error}"
        else:
            label = f"✓ {name}  ({ms:.1f}ms)"

        # Store full step data in node data for detail panel
        if step.get("children"):
            node = parent.add(label, data=step)
            for child in step["children"]:
                self._add_step(node, child)
        else:
            parent.add_leaf(label, data=step)
