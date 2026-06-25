"""
Telemetry & Observability Screen — metrics, cache, hooks, trace comparison.

Features:
  • Per-operation latency sparklines
  • Performance percentile table (P50/P95/P99)
  • Cost breakdown table
  • Telemetry status (OTel / Prometheus)
  • Embedding cache stats
  • Hook management
  • Trace comparison
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, DataTable, Sparkline, RichLog, Button, Input

from glassbox_rag.tui.widgets.collapsible_panel import CollapsiblePanel


class TelemetryScreen(Container):
    """Performance monitoring, telemetry, and observability dashboard."""

    DEFAULT_CSS = """
    TelemetryScreen {
        height: 100%;
        padding: 1 0;
        overflow-y: auto;
    }
    #tele-title {
        color: #c4b5fd;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #tele-sub {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #tele-top {
        layout: horizontal;
        height: auto;
        min-height: 8;
        margin-bottom: 1;
    }
    .tele-spark-panel {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1 0 0;
    }
    .tele-spark-panel:last-child {
        margin-right: 0;
    }
    .spark-title {
        color: #8b8fa3;
        margin-bottom: 1;
    }
    #tele-tables {
        layout: horizontal;
        height: auto;
        min-height: 10;
        margin-bottom: 1;
    }
    #perf-panel {
        width: 2fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin-right: 1;
    }
    #cost-panel {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    .table-title {
        color: #c4b5fd;
        text-style: bold;
        margin-bottom: 1;
    }
    #cache-stats-body {
        height: auto;
    }
    #hooks-body {
        height: auto;
    }
    #hooks-table {
        height: auto;
        max-height: 8;
    }
    #trace-cmp-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #trace-a-input, #trace-b-input {
        width: 1fr;
        margin-right: 1;
    }
    #compare-btn {
        width: auto;
        min-width: 16;
    }
    #compare-output {
        height: auto;
        max-height: 12;
        min-height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("⬡  Telemetry & Observability", id="tele-title")
        yield Static("Metrics, cache, hooks, and trace analysis", id="tele-sub")

        # ── Sparklines ────────────────────────────────────────────
        with Horizontal(id="tele-top"):
            with Vertical(classes="tele-spark-panel"):
                yield Static("Latency (ms)", classes="spark-title")
                yield Sparkline([0.0], id="spark-latency", summary_function=max)

            with Vertical(classes="tele-spark-panel"):
                yield Static("Requests", classes="spark-title")
                yield Sparkline([0.0], id="spark-requests", summary_function=max)

            with Vertical(classes="tele-spark-panel"):
                yield Static("Tokens", classes="spark-title")
                yield Sparkline([0.0], id="spark-tokens", summary_function=max)

            with Vertical(classes="tele-spark-panel"):
                yield Static("Cost ($)", classes="spark-title")
                yield Sparkline([0.0], id="spark-cost", summary_function=max)

        # ── Performance + Cost tables ─────────────────────────────
        with Horizontal(id="tele-tables"):
            with Vertical(id="perf-panel"):
                yield Static("Performance Metrics", classes="table-title")
                yield DataTable(id="perf-table", cursor_type="row")

            with Vertical(id="cost-panel"):
                yield Static("Cost Breakdown", classes="table-title")
                yield DataTable(id="cost-table", cursor_type="row")

        # ── Collapsible: Telemetry Status ─────────────────────────
        with CollapsiblePanel(title="Telemetry Status (OTel / Prometheus)", collapsed=True):
            with Horizontal():
                yield Vertical(id="otel-status")
                yield Vertical(id="prom-status")

        # ── Collapsible: Cache Stats ──────────────────────────────
        with CollapsiblePanel(title="Embedding Cache Statistics", collapsed=False):
            yield Vertical(id="cache-stats-body")

        # ── Collapsible: Hooks ────────────────────────────────────
        with CollapsiblePanel(title="Lifecycle Hooks", collapsed=True):
            yield Vertical(id="hooks-body")

        # ── Collapsible: Trace Comparison ─────────────────────────
        with CollapsiblePanel(title="Trace Comparison", collapsed=True):
            with Horizontal(id="trace-cmp-row"):
                yield Input(placeholder="Trace ID A", id="trace-a-input")
                yield Input(placeholder="Trace ID B", id="trace-b-input")
                yield Button("⚖  Compare", id="compare-btn", variant="primary")
            yield RichLog(id="compare-output", highlight=True, markup=True, max_lines=100)

    def on_mount(self) -> None:
        perf = self.query_one("#perf-table", DataTable)
        perf.add_column("Operation", width=18, key="op")
        perf.add_column("P50", width=10, key="p50")
        perf.add_column("P95", width=10, key="p95")
        perf.add_column("P99", width=10, key="p99")
        perf.add_column("Mean", width=10, key="mean")
        perf.add_column("Count", width=8, key="count")

        cost = self.query_one("#cost-table", DataTable)
        cost.add_column("Item", width=22, key="item")
        cost.add_column("Cost", width=14, key="cost")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare-btn":
            await self._run_comparison()

    async def refresh_data(self, client) -> None:
        metrics = await client.get_metrics()

        # ── Sparklines ────────────────────────────────────────────
        try:
            self.query_one("#spark-latency", Sparkline).data = client.get_sparkline_data("latency")
            self.query_one("#spark-requests", Sparkline).data = client.get_sparkline_data("requests")
            self.query_one("#spark-tokens", Sparkline).data = client.get_sparkline_data("tokens")
            self.query_one("#spark-cost", Sparkline).data = client.get_sparkline_data("cost")
        except Exception:
            pass

        # ── Performance table ─────────────────────────────────────
        perf = self.query_one("#perf-table", DataTable)
        perf.clear()
        if metrics.histograms:
            for op, h in metrics.histograms.items():
                perf.add_row(
                    op,
                    f"{h.get('p50_ms', h.get('mean_ms', 0)):.0f}ms",
                    f"{h.get('p95_ms', 0):.0f}ms",
                    f"{h.get('p99_ms', 0):.0f}ms",
                    f"{h.get('mean_ms', 0):.1f}ms",
                    str(h.get("count", 0)),
                )
        else:
            perf.add_row("📊", "No data", "—", "—", "Run queries to populate", "0")

        # ── Cost table ────────────────────────────────────────────
        cost_table = self.query_one("#cost-table", DataTable)
        cost_table.clear()
        cost_table.add_row("Total Pipeline", f"${metrics.total_cost_usd:.6f}")
        if metrics.histograms:
            for op, h in metrics.histograms.items():
                c = h.get("cost_usd", 0)
                if c:
                    cost_table.add_row(op, f"${c:.6f}")

        # ── Telemetry status ──────────────────────────────────────
        tel_status = client.get_telemetry_status()

        otel_panel = self.query_one("#otel-status", Vertical)
        await otel_panel.remove_children()
        otel_on = tel_status.get("otel_enabled", False)
        otel_dot = "🟢" if otel_on else "⚪"
        backend = tel_status.get("otel_backend", "none")
        await otel_panel.mount(Static(f"{otel_dot}  [bold]OpenTelemetry[/]  {'active' if otel_on else 'disabled'}"))
        if otel_on:
            await otel_panel.mount(Static(f"    Backend: {backend}"))
            endpoint = tel_status.get("otel_endpoint", "")
            if endpoint:
                await otel_panel.mount(Static(f"    Endpoint: {endpoint}"))

        prom_panel = self.query_one("#prom-status", Vertical)
        await prom_panel.remove_children()
        prom_on = tel_status.get("prometheus_enabled", False)
        prom_dot = "🟢" if prom_on else "⚪"
        await prom_panel.mount(Static(f"{prom_dot}  [bold]Prometheus[/]  {'active' if prom_on else 'disabled'}"))
        if prom_on:
            port = tel_status.get("prometheus_port", "")
            await prom_panel.mount(Static(f"    Port: {port}"))

        # ── Cache stats ───────────────────────────────────────────
        cache_body = self.query_one("#cache-stats-body", Vertical)
        await cache_body.remove_children()
        cache = client.get_cache_stats()
        if cache:
            hits = cache.get("hits", 0)
            misses = cache.get("misses", 0)
            total = hits + misses
            ratio = f"{(hits / total * 100):.1f}%" if total > 0 else "N/A"
            size = cache.get("size", cache.get("current_size", 0))
            max_size = cache.get("max_size", cache.get("capacity", "∞"))

            await cache_body.mount(Static(f"  [#c4b5fd]Hit Rate:[/]     {ratio}  ({hits} hits / {misses} misses)"))
            await cache_body.mount(Static(f"  [#c4b5fd]Lookups:[/]      {total}"))
            await cache_body.mount(Static(f"  [#c4b5fd]Cache Size:[/]   {size} / {max_size}"))
            await cache_body.mount(Static(f"  [#c4b5fd]Evictions:[/]    {cache.get('evictions', 0)}"))
        else:
            await cache_body.mount(Static("[#5c5f73]  📭  No cache data available[/]"))

        # ── Hooks info ────────────────────────────────────────────
        hooks_body = self.query_one("#hooks-body", Vertical)
        await hooks_body.remove_children()
        hook_info = client.get_hook_info()
        if hook_info:
            total_hooks = hook_info.get("total_hooks", 0)
            await hooks_body.mount(Static(f"  [#c4b5fd]Total Hooks:[/]  {total_hooks}"))
            hooks_list = hook_info.get("hooks", {})
            if isinstance(hooks_list, dict):
                for point, entries in hooks_list.items():
                    if entries:
                        await hooks_body.mount(Static(f"  [#8b8fa3]{point}:[/]"))
                        for h in entries:
                            name = h.get("name", "?") if isinstance(h, dict) else str(h)
                            priority = h.get("priority", "—") if isinstance(h, dict) else "—"
                            await hooks_body.mount(Static(f"    • {name}  (priority: {priority})"))
            elif isinstance(hooks_list, list):
                for h in hooks_list:
                    name = h.get("name", "?") if isinstance(h, dict) else str(h)
                    await hooks_body.mount(Static(f"    • {name}"))
            if total_hooks == 0:
                await hooks_body.mount(Static("[#5c5f73]  No hooks registered[/]"))
        else:
            await hooks_body.mount(Static("[#5c5f73]  📭  No hook data available[/]"))

    async def _run_comparison(self) -> None:
        output = self.query_one("#compare-output", RichLog)
        output.clear()

        app = self.app
        client = getattr(app, "client", None)
        if not client or not client.ready:
            output.write("[red]✗ Engine not ready[/]")
            return

        id_a = self.query_one("#trace-a-input", Input).value.strip()
        id_b = self.query_one("#trace-b-input", Input).value.strip()

        if not id_a or not id_b:
            output.write("[yellow]⚠ Enter two trace IDs to compare[/]")
            return

        output.write(f"[#8b8fa3]Comparing {id_a[:12]}… vs {id_b[:12]}…[/]")

        result = await client.compare_traces(id_a, id_b)
        if not result:
            output.write("[red]✗ Could not find one or both traces[/]")
            return

        output.write(f"[bold #c4b5fd]Trace Comparison[/]")
        output.write("")

        # Duration comparison
        dur_a = result.get("trace_a_duration_ms", 0)
        dur_b = result.get("trace_b_duration_ms", 0)
        delta = dur_b - dur_a
        delta_color = "#34d399" if delta < 0 else "#f43f5e" if delta > 0 else "#8b8fa3"
        output.write(f"  Duration A: {dur_a:.1f}ms")
        output.write(f"  Duration B: {dur_b:.1f}ms")
        output.write(f"  Delta:      [{delta_color}]{delta:+.1f}ms[/]")
        output.write("")

        # Step comparisons
        step_diffs = result.get("step_diffs", [])
        if step_diffs:
            output.write("[bold]Step-by-step:[/]")
            for sd in step_diffs:
                name = sd.get("step", "?")
                a_ms = sd.get("a_ms", 0)
                b_ms = sd.get("b_ms", 0)
                diff = b_ms - a_ms
                color = "#34d399" if diff < 0 else "#f43f5e" if diff > 0 else "#8b8fa3"
                output.write(
                    f"  {name:<20} {a_ms:>7.1f}ms → {b_ms:>7.1f}ms  "
                    f"[{color}]({diff:+.1f}ms)[/]"
                )
