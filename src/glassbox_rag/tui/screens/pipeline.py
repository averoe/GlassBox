"""
Pipeline Screen — interactive RAG pipeline visualisation.

Features:
  • ASCII pipeline diagram with animation
  • Drag-and-drop step reordering (Alt+↑/↓)
  • Test query execution with live step timing
  • Step detail panel
"""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Input, Button, RichLog

from glassbox_rag.tui.widgets.pipeline_view import PipelineView
from glassbox_rag.tui.widgets.drag_list import DragList


_PIPELINE_STEPS = [
    ("input", "Query Input", "📝"),
    ("encode", "Encode Query", "🧠"),
    ("retrieve", "Retrieve Documents", "🔍"),
    ("rerank", "Rerank Results", "🔄"),
    ("generate", "Generate Response", "⚡"),
    ("output", "Final Output", "📤"),
]

_STEP_DESCRIPTIONS = {
    "input": "Accepts and validates user queries for processing.",
    "encode": "Converts text queries to dense vector embeddings using the configured encoder.",
    "retrieve": "Searches the vector store for semantically relevant documents.",
    "rerank": "Reorders retrieval results using a cross-encoder for improved precision.",
    "generate": "Produces a natural-language answer using retrieved context and an LLM.",
    "output": "Formats and returns the final response to the user.",
}


class PipelineScreen(Container):
    """Interactive pipeline visualisation and testing."""

    DEFAULT_CSS = """
    PipelineScreen {
        height: 100%;
        padding: 1 0;
    }
    #pipeline-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #pipeline-sub {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #pipeline-controls {
        layout: horizontal;
        height: auto;
        margin: 1 0;
    }
    #query-input {
        width: 2fr;
        margin-right: 1;
    }
    #run-btn {
        width: auto;
        min-width: 18;
        margin-right: 1;
    }
    #reset-btn {
        width: auto;
        min-width: 12;
        background: #22263a;
        border: tall #2a2e42;
    }
    #pipeline-bottom {
        layout: horizontal;
        height: 1fr;
        min-height: 14;
        margin-top: 1;
    }
    #step-detail {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin-right: 1;
    }
    #execution-timeline {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    #drag-section {
        margin-top: 1;
        height: auto;
    }
    #drag-label {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    #drag-hint {
        color: #5c5f73;
        text-style: italic;
        margin-top: 1;
    }
    .detail-title {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("⬡  Pipeline Visualization", id="pipeline-title")
        yield Static("Interactive view of your RAG pipeline execution flow", id="pipeline-sub")

        yield PipelineView(id="pipeline-view")

        with Horizontal(id="pipeline-controls"):
            yield Input(placeholder="Enter a test query…", id="query-input")
            yield Button("▶  Run Test Query", id="run-btn")
            yield Button("↺  Reset", id="reset-btn", classes="-secondary")

        with Horizontal(id="pipeline-bottom"):
            with Vertical(id="step-detail"):
                yield Static("Step Details", classes="detail-title")
                yield Static("[dim italic]Click a pipeline step or run a query[/]", id="step-detail-content")

            with Vertical(id="execution-timeline"):
                yield Static("Execution Timeline", classes="detail-title")
                yield RichLog(id="timeline-log", highlight=True, markup=True, max_lines=30)

        with Vertical(id="drag-section"):
            yield Static("Pipeline Step Order  (Alt+↑/↓ to reorder)", id="drag-label")
            yield DragList(
                items=[(sid, label, icon) for sid, label, icon in _PIPELINE_STEPS],
                id="step-drag-list",
            )
            yield Static("Drag steps to customise pipeline execution order", id="drag-hint")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            await self._run_test_query()
        elif event.button.id == "reset-btn":
            self._reset()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            await self._run_test_query()

    async def _run_test_query(self) -> None:
        inp = self.query_one("#query-input", Input)
        query = inp.value.strip() or "What is GlassBox RAG?"
        pipeline = self.query_one("#pipeline-view", PipelineView)
        timeline = self.query_one("#timeline-log", RichLog)
        detail = self.query_one("#step-detail-content", Static)

        timeline.clear()
        timeline.write(f"[bold]Query:[/] {query}")
        timeline.write("")

        # Animate steps
        step_ids = [sid for sid, _, _ in _PIPELINE_STEPS]
        for sid in step_ids:
            pipeline.set_active(sid)
            timeline.write(f"  ⏳ {sid}…")
            await asyncio.sleep(0.15)

        # Run the actual retrieve
        app = self.app
        client = getattr(app, "client", None)
        if client and client.ready:
            result = await client.run_retrieve(query, top_k=3)
            if result.get("error"):
                timeline.write(f"\n[red]Error: {result['error']}[/]")
                pipeline.set_active(None)
                return

            exec_ms = result.get("execution_time_ms", 0)
            n = result.get("num_results", 0)
            strategy = result.get("strategy", "?")
            trace_id = result.get("trace_id", "")

            # Show timing per step from trace
            timings: dict[str, float] = {}
            if trace_id:
                tv = await client.get_trace_visualization(trace_id)
                if tv and tv.get("waterfall"):
                    # Parse real step timings from waterfall data
                    _step_name_map = {
                        "encode_query": "encode",
                        "adaptive_retrieve": "retrieve",
                        "rerank": "rerank",
                        "generate": "generate",
                        "retrieve": "input",
                    }
                    for step in tv["waterfall"]:
                        step_name = step.get("name", "")
                        mapped = _step_name_map.get(step_name, step_name)
                        if mapped in {s[0] for s in _PIPELINE_STEPS}:
                            timings[mapped] = step.get("duration_ms", 0)
                    # Fill input/output with small fractions if not in waterfall
                    if "input" not in timings:
                        timings["input"] = exec_ms * 0.02
                    if "output" not in timings:
                        timings["output"] = exec_ms * 0.03

            # Fallback to proportional estimates if no waterfall data
            if not timings:
                timings = {
                    "input": exec_ms * 0.02,
                    "encode": exec_ms * 0.15,
                    "retrieve": exec_ms * 0.55,
                    "rerank": exec_ms * 0.12,
                    "generate": exec_ms * 0.10,
                    "output": exec_ms * 0.06,
                }

            pipeline.set_timings(timings)
            pipeline.set_active(None)

            timeline.write("")
            timeline.write("[bold green]✓ Execution complete[/]")
            timeline.write(f"  Strategy:     {strategy}")
            timeline.write(f"  Results:      {n}")
            timeline.write(f"  Total:        [bold]{exec_ms:.1f}ms[/]")
            timeline.write(f"  Trace:        {trace_id[:12]}…" if trace_id else "")

            detail.update(
                f"[bold]Last Query Result[/]\n\n"
                f"Strategy: {strategy}\n"
                f"Results:  {n}\n"
                f"Time:     {exec_ms:.1f}ms\n"
                f"Trace:    {trace_id[:12] if trace_id else 'N/A'}\n\n"
                + "\n".join(
                    f"  [{i+1}] (score={d.get('score',0):.3f}) {d.get('content','')[:80]}…"
                    for i, d in enumerate(result.get("documents", []))
                )
            )
        else:
            pipeline.set_active(None)
            timeline.write("\n[yellow]Engine not initialised — showing demo animation only[/]")

    def _reset(self) -> None:
        self.query_one("#pipeline-view", PipelineView).reset()
        self.query_one("#timeline-log", RichLog).clear()
        self.query_one("#step-detail-content", Static).update(
            "[dim italic]Click a pipeline step or run a query[/]"
        )

    async def refresh_data(self, client) -> None:
        """Lightweight refresh — pipeline view is mostly user-driven."""
        pass
