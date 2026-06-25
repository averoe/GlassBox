"""
Debugger Screen — trace inspection, filtering, anomaly detection.

Features:
  • Filterable trace list (All / Errors / Slow)
  • Hierarchical trace step tree
  • Step detail panel with input/output data
  • Trace export (JSON)
  • Anomaly detection
  • Trace comparison
"""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, Select, Input, RichLog, DataTable

from glassbox_rag.tui.widgets.trace_tree import TraceTree
from glassbox_rag.tui.widgets.collapsible_panel import CollapsiblePanel


class DebuggerScreen(Container):
    """Trace debugger and analysis panel."""

    DEFAULT_CSS = """
    DebuggerScreen {
        height: 100%;
        padding: 1 0;
    }
    #debug-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #debug-controls {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #trace-filter {
        width: 18;
        margin-right: 1;
    }
    #trace-search {
        width: 1fr;
        margin-right: 1;
    }
    #anomaly-btn {
        width: auto;
        min-width: 18;
        background: #22263a;
        border: tall #2a2e42;
        margin-right: 1;
    }
    #export-btn {
        width: auto;
        min-width: 14;
        background: #22263a;
        border: tall #2a2e42;
    }
    #debug-panels {
        layout: horizontal;
        height: 1fr;
    }
    #trace-list-panel {
        width: 2fr;
        background: #181b27;
        border: round #2a2e42;
        margin-right: 1;
        padding: 1;
    }
    #trace-detail-panel {
        width: 3fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1;
    }
    #trace-table {
        height: 100%;
    }
    #anomaly-output {
        height: auto;
        max-height: 10;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._traces: list[dict] = []
        self._selected_trace_id: str | None = None
        self._filter = "all"

    def compose(self) -> ComposeResult:
        yield Static("⬡  Trace Debugger", id="debug-title")

        with Horizontal(id="debug-controls"):
            yield Select(
                [("All Traces", "all"), ("Errors Only", "errors"), ("Slow (>1s)", "slow")],
                value="all",
                id="trace-filter",
            )
            yield Input(placeholder="Search traces…", id="trace-search")
            yield Button("🔎 Anomalies", id="anomaly-btn", classes="-secondary")
            yield Button("💾 Export", id="export-btn", classes="-secondary")

        with Horizontal(id="debug-panels"):
            with Vertical(id="trace-list-panel"):
                yield DataTable(id="trace-table", cursor_type="row")

            with Vertical(id="trace-detail-panel"):
                with CollapsiblePanel(title="Trace Details", collapsed=False):
                    yield RichLog(id="trace-detail-log", highlight=True, markup=True, max_lines=200)
                with CollapsiblePanel(title="Step Tree", collapsed=False):
                    yield TraceTree(id="trace-tree")

        with CollapsiblePanel(title="Anomaly Detection", collapsed=True, id="anomaly-panel"):
            yield Vertical(id="anomaly-output")

    def on_mount(self) -> None:
        table = self.query_one("#trace-table", DataTable)
        table.add_column("ID", width=12, key="id")
        table.add_column("Operation", width=16, key="op")
        table.add_column("Duration", width=10, key="dur")
        table.add_column("Status", width=8, key="status")
        table.add_column("Time", width=20, key="time")

        # Styled empty state for trace detail
        log = self.query_one("#trace-detail-log", RichLog)
        log.write("[#5c5f73]  🔍  Select a trace from the list to view details[/]")
        log.write("[#5c5f73]     or run a query to generate new traces.[/]")

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "trace-filter":
            self._filter = str(event.value)
            self._render_trace_table()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "trace-search":
            self._render_trace_table()

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one("#trace-table", DataTable)
        try:
            row_data = table.get_row(event.row_key)
            trace_id_short = str(row_data[0]).replace("…", "")
            # Find the full trace_id
            for t in self._traces:
                tid = t.get("id", t.get("trace_id", ""))
                if tid.startswith(trace_id_short):
                    self._selected_trace_id = tid
                    await self._show_trace_detail(t)
                    break
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "anomaly-btn":
            await self._run_anomaly_detection()
        elif event.button.id == "export-btn":
            self._export_selected()

    def _render_trace_table(self) -> None:
        table = self.query_one("#trace-table", DataTable)
        table.clear()

        search = ""
        try:
            search = self.query_one("#trace-search", Input).value.lower()
        except Exception:
            pass

        filtered = self._filtered_traces()

        # Styled empty state
        if not filtered:
            from rich.text import Text
            table.add_row(
                "—",
                Text("📭  No traces yet", style="#5c5f73 italic"),
                "—",
                Text("—", style="#5c5f73"),
                "—",
            )
            return

        from rich.text import Text

        for t in filtered:
            tid = t.get("id", t.get("trace_id", ""))[:10] + "…"
            root = t.get("root_step", {})
            op = root.get("name", "request") if root else "request"
            dur_ms = t.get("duration_ms", 0)
            dur = f"{dur_ms:.0f}ms"
            err = root.get("error") if root else None
            ts = t.get("start_time", "")[:19]

            # Color-code by status
            if err:
                status = Text("❌ error", style="#f43f5e")
                dur_text = Text(dur, style="#f43f5e")
            elif dur_ms > 1000:
                status = Text("⚠ slow", style="#fbbf24")
                dur_text = Text(dur, style="#fbbf24 bold")
            else:
                status = Text("✓ ok", style="#34d399")
                dur_text = Text(dur, style="#e8eaf0")

            # Search filter
            query_text = ""
            if root:
                inp = root.get("input_data", {})
                query_text = inp.get("query", "").lower() if isinstance(inp, dict) else ""

            if search and search not in op.lower() and search not in query_text and search not in tid.lower():
                continue

            table.add_row(tid, op, dur_text, status, ts)

    def _filtered_traces(self) -> list[dict]:
        if self._filter == "errors":
            return [t for t in self._traces
                    if t.get("root_step", {}).get("error")]
        elif self._filter == "slow":
            return [t for t in self._traces if t.get("duration_ms", 0) > 1000]
        return self._traces

    async def _show_trace_detail(self, trace: dict) -> None:
        log = self.query_one("#trace-detail-log", RichLog)
        tree = self.query_one("#trace-tree", TraceTree)

        log.clear()
        tid = trace.get("id", trace.get("trace_id", ""))
        dur = trace.get("duration_ms", 0)
        n_steps = trace.get("num_steps", 0)
        req_id = trace.get("request_id", "")

        log.write(f"[bold]Trace ID:[/]    {tid}")
        log.write(f"[bold]Request ID:[/]  {req_id}")
        log.write(f"[bold]Duration:[/]    {dur:.1f}ms")
        log.write(f"[bold]Steps:[/]       {n_steps}")
        log.write(f"[bold]Start:[/]       {trace.get('start_time', '')}")
        log.write(f"[bold]End:[/]         {trace.get('end_time', '')}")

        root = trace.get("root_step", {})
        if root:
            err = root.get("error")
            if err:
                log.write(f"\n[red bold]Error:[/] {err}")

            # Show input/output for root
            inp = root.get("input_data", {})
            if inp:
                log.write(f"\n[bold #8b5cf6]Input Data:[/]")
                for k, v in inp.items():
                    log.write(f"  {k}: {v}")

            out = root.get("output_data", {})
            if out:
                log.write(f"\n[bold #22d3ee]Output Data:[/]")
                for k, v in out.items():
                    log.write(f"  {k}: {v}")

        # Load the trace tree
        tree.load_trace(trace)

    async def _run_anomaly_detection(self) -> None:
        app = self.app
        client = getattr(app, "client", None)
        panel = self.query_one("#anomaly-output", Vertical)
        await panel.remove_children()

        if not client or not client.ready:
            await panel.mount(Static("[yellow]Engine not ready[/]"))
            return

        result = await client.get_anomalies()
        anomalies = result.get("anomalies", [])
        count = result.get("traces_analyzed", 0)

        await panel.mount(Static(
            f"[bold #8b5cf6]Anomaly Detection[/]  — {count} traces analysed, "
            f"{len(anomalies)} anomalies found"
        ))

        if not anomalies:
            await panel.mount(Static("[green]No anomalies detected ✓[/]"))
        else:
            for a in anomalies[:10]:
                step = a.get("step", "?")
                dur = a.get("duration_ms", 0)
                mean = a.get("mean_ms", 0)
                factor = a.get("times_slower", 0)
                await panel.mount(Static(
                    f"  [red]⚠[/] {step:<20} {dur:.1f}ms  "
                    f"(mean {mean:.1f}ms, {factor:.1f}x slower)"
                ))

    def _export_selected(self) -> None:
        if not self._selected_trace_id:
            return
        for t in self._traces:
            if t.get("id", t.get("trace_id", "")) == self._selected_trace_id:
                data = json.dumps(t, indent=2, default=str)
                from pathlib import Path
                export_dir = Path("data") / "exports"
                export_dir.mkdir(parents=True, exist_ok=True)
                path = export_dir / f"trace_{self._selected_trace_id[:8]}.json"
                try:
                    path.write_text(data, encoding="utf-8")
                    self.notify(f"Trace exported to {path}", title="Export")
                except Exception as e:
                    self.notify(f"Export failed: {e}", title="Error", severity="error")
                return

    async def refresh_data(self, client) -> None:
        self._traces = await client.get_traces(limit=50)
        self._render_trace_table()
