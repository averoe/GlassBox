"""
PipelineView widget — animated ASCII RAG pipeline with color-coding.
"""

from __future__ import annotations

from textual.widgets import Static


_STEPS = [
    ("input", "📝 Query"),
    ("encode", "🧠 Encode"),
    ("retrieve", "🔍 Retrieve"),
    ("rerank", "🔄 Rerank"),
    ("generate", "⚡ Generate"),
    ("output", "📤 Output"),
]


def _build_pipeline_ascii(
    active_step: str | None = None,
    timings: dict[str, float] | None = None,
    completed: set[str] | None = None,
    error_steps: set[str] | None = None,
) -> str:
    """Build a Rich-markup colored ASCII pipeline string."""
    timings = timings or {}
    completed = completed or set()
    error_steps = error_steps or set()
    top = ""
    mid = ""
    bot = ""
    dur = ""

    for i, (sid, label) in enumerate(_STEPS):
        name = label.center(14)
        is_active = sid == active_step
        is_error = sid in error_steps
        is_done = sid in completed

        # Color selection: active=indigo glow, error=red, done=green, default=dim
        if is_active:
            color = "#6366f1"
            top += f"[{color}]╔══════════════╗[/]"
            mid += f"[{color} bold]║{name}║[/]"
            bot += f"[{color}]╚══════════════╝[/]"
        elif is_error:
            color = "#f43f5e"
            top += f"[{color}]┌──────────────┐[/]"
            mid += f"[{color}]│{name}│[/]"
            bot += f"[{color}]└──────────────┘[/]"
        elif is_done:
            color = "#34d399"
            top += f"[{color}]┌──────────────┐[/]"
            mid += f"[{color}]│{name}│[/]"
            bot += f"[{color}]└──────────────┘[/]"
        else:
            top += "[#5c5f73]┌──────────────┐[/]"
            mid += f"[#8b8fa3]│{name}│[/]"
            bot += "[#5c5f73]└──────────────┘[/]"

        t = timings.get(sid)
        time_str = f"{t:.1f}ms" if t is not None else ""
        dur += time_str.center(16)

        if i < len(_STEPS) - 1:
            if is_active or is_done:
                top += "    "
                mid += "[#6366f1]───▶[/]"
                bot += "    "
            else:
                top += "    "
                mid += "[#5c5f73]───▶[/]"
                bot += "    "
            dur += "    "

    return f"{top}\n{mid}\n{bot}\n[#8b8fa3]{dur}[/]"


class PipelineView(Static):
    """Renders a color-coded ASCII pipeline diagram."""

    DEFAULT_CSS = """
    PipelineView {
        height: auto;
        min-height: 6;
        background: #12141c;
        border: round #2a2e42;
        padding: 1 2;
        color: #e8eaf0;
        overflow-x: auto;
    }
    PipelineView:hover {
        border: round #6366f1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(_build_pipeline_ascii(), markup=True, **kwargs)
        self._active: str | None = None
        self._timings: dict[str, float] = {}
        self._completed: set[str] = set()
        self._errors: set[str] = set()

    def set_active(self, step_id: str | None) -> None:
        """Set the currently active step and mark previous ones as complete."""
        if self._active and step_id != self._active:
            self._completed.add(self._active)
        self._active = step_id
        self._refresh_display()

    def set_timings(self, timings: dict[str, float]) -> None:
        self._timings = timings
        # Mark all steps with timings as completed
        self._completed.update(timings.keys())
        self._refresh_display()

    def set_error(self, step_id: str) -> None:
        """Mark a step as errored."""
        self._errors.add(step_id)
        self._refresh_display()

    def reset(self) -> None:
        self._active = None
        self._timings = {}
        self._completed = set()
        self._errors = set()
        self._refresh_display()

    def _refresh_display(self) -> None:
        self.update(_build_pipeline_ascii(
            active_step=self._active,
            timings=self._timings,
            completed=self._completed,
            error_steps=self._errors,
        ))
