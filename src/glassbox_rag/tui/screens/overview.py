"""
Overview Screen — live system dashboard.

Shows branded splash banner, metric cards with sparklines,
component health, recent activity feed, and system warnings.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, DataTable, RichLog
from glassbox_rag.tui.widgets.metric_card import MetricCard
from glassbox_rag.tui.widgets.status_indicator import StatusIndicator
from glassbox_rag.tui.widgets.collapsible_panel import CollapsiblePanel


# ── Branded ASCII Banner ──────────────────────────────────────────

_BANNER = (
    "[#c4b5fd]   ██████╗ ██╗      █████╗ ███████╗███████╗[/]"
    "[#d4c5ff]██████╗  ██████╗ ██╗  ██╗[/]\n"
    "[#c4b5fd]  ██╔════╝ ██║     ██╔══██╗██╔════╝██╔════╝[/]"
    "[#d4c5ff]██╔══██╗██╔═══██╗╚██╗██╔╝[/]\n"
    "[#c4b5fd]  ██║  ███╗██║     ███████║███████╗███████╗[/]"
    "[#d4c5ff]██████╔╝██║   ██║ ╚███╔╝[/]\n"
    "[#c4b5fd]  ██║   ██║██║     ██╔══██║╚════██║╚════██║[/]"
    "[#d4c5ff]██╔══██╗██║   ██║ ██╔██╗[/]\n"
    "[#c4b5fd]  ╚██████╔╝███████╗██║  ██║███████║███████║[/]"
    "[#d4c5ff]██████╔╝╚██████╔╝██╔╝ ██╗[/]\n"
    "[#c4b5fd]   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝[/]"
    "[#d4c5ff]╚═════╝  ╚═════╝╚═╝  ╚═╝[/]\n"
    "\n"
    "[#c4b5fd]             ██████╗  █████╗  ██████╗[/]\n"
    "[#d4c5ff]             ██╔══██╗██╔══██╗██╔════╝[/]\n"
    "[#c4b5fd]             ██████╔╝███████║██║  ███╗[/]\n"
    "[#d4c5ff]             ██╔══██╗██╔══██║██║   ██║[/]\n"
    "[#c4b5fd]             ██║  ██║██║  ██║╚██████╔╝[/]\n"
    "[#d4c5ff]             ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝[/]"
)


class OverviewScreen(Container):
    """Main overview dashboard panel."""

    DEFAULT_CSS = """
    OverviewScreen {
        height: 100%;
        padding: 1 0;
    }
    #brand-banner {
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    #overview-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #overview-subtitle {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #metric-row {
        layout: horizontal;
        height: auto;
        min-height: 8;
        margin: 0 0 1 0;
    }
    #health-activity {
        layout: horizontal;
        height: 1fr;
        min-height: 12;
    }
    #health-panel {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1 0 0;
    }
    #health-panel:hover {
        border: round #6366f1;
    }
    #activity-panel {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    #activity-panel:hover {
        border: round #6366f1;
    }
    .panel-title {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._banner_visible = True

    def compose(self) -> ComposeResult:
        yield Static(_BANNER, id="brand-banner")
        yield Static("⬡  System Overview", id="overview-title")
        yield Static("Real-time status of your GlassBox RAG pipeline", id="overview-subtitle")

        with Horizontal(id="metric-row"):
            yield MetricCard(
                label="Total Requests", value="0", icon="⚡",
                spark_color="#6366f1", id="mc-requests",
            )
            yield MetricCard(
                label="Avg Latency", value="0ms", icon="⏱",
                spark_color="#22d3ee", id="mc-latency",
            )
            yield MetricCard(
                label="Tokens Used", value="0", icon="🔤",
                spark_color="#34d399", id="mc-tokens",
            )
            yield MetricCard(
                label="Total Cost", value="$0.00", icon="💰",
                spark_color="#fbbf24", id="mc-cost",
            )

        with Horizontal(id="health-activity"):
            with Vertical(id="health-panel"):
                yield Static("Components", classes="panel-title")
                yield Vertical(id="component-list")

            with Vertical(id="activity-panel"):
                yield Static("Recent Activity", classes="panel-title")
                yield RichLog(id="activity-log", highlight=True, markup=True, max_lines=50)

        with CollapsiblePanel(title="Warnings", collapsed=False, id="warnings-cp"):
            yield Vertical(id="warnings-panel")

    async def on_mount(self) -> None:
        """Collapse the banner after a few data refreshes to save space."""
        self._refresh_count = 0
        self._apply_responsive_layout()

    def on_resize(self, event) -> None:
        """Switch metric cards to vertical layout on narrow terminals."""
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        """Apply responsive CSS rules based on current container width."""
        try:
            width = self.size.width
            metric_row = self.query_one("#metric-row")
            health_row = self.query_one("#health-activity")

            if width < 80:
                metric_row.styles.layout = "vertical"
                health_row.styles.layout = "vertical"
            else:
                metric_row.styles.layout = "horizontal"
                health_row.styles.layout = "horizontal"
        except Exception:
            pass

    async def refresh_data(self, client) -> None:
        """Pull fresh data from the engine client and update all widgets."""
        # Collapse banner after 3 refreshes
        self._refresh_count = getattr(self, "_refresh_count", 0) + 1
        if self._refresh_count >= 3 and self._banner_visible:
            try:
                banner = self.query_one("#brand-banner", Static)
                banner.display = False
                self._banner_visible = False
            except Exception:
                pass

        health = await client.get_health()
        metrics = await client.get_metrics()
        traces = await client.get_traces(limit=10)

        # ── Metrics cards ─────────────────────────────────────────
        prev_req = getattr(self, "_prev_req", 0)
        prev_lat = getattr(self, "_prev_lat", 0.0)
        prev_tok = getattr(self, "_prev_tok", 0)
        prev_cost = getattr(self, "_prev_cost", 0.0)

        def _trend(curr, prev):
            if prev == 0:
                return "+0%", True
            pct = ((curr - prev) / prev) * 100
            return f"{'+'if pct >= 0 else ''}{pct:.1f}%", pct >= 0

        req_trend, req_up = _trend(metrics.total_requests, prev_req)
        lat_trend, lat_up = _trend(metrics.avg_latency_ms, prev_lat)
        tok_trend, tok_up = _trend(metrics.total_tokens, prev_tok)
        cost_trend, cost_up = _trend(metrics.total_cost_usd, prev_cost)

        try:
            self.query_one("#mc-requests", MetricCard).update_data(
                f"{metrics.total_requests:,}", req_trend, req_up,
                client.get_sparkline_data("requests"),
            )
            self.query_one("#mc-latency", MetricCard).update_data(
                f"{metrics.avg_latency_ms:.0f}ms", lat_trend, not lat_up,
                client.get_sparkline_data("latency"),
            )
            self.query_one("#mc-tokens", MetricCard).update_data(
                f"{metrics.total_tokens:,}", tok_trend, tok_up,
                client.get_sparkline_data("tokens"),
            )
            self.query_one("#mc-cost", MetricCard).update_data(
                f"${metrics.total_cost_usd:.4f}", cost_trend, not cost_up,
                client.get_sparkline_data("cost"),
            )
        except Exception:
            pass

        self._prev_req = metrics.total_requests
        self._prev_lat = metrics.avg_latency_ms
        self._prev_tok = metrics.total_tokens
        self._prev_cost = metrics.total_cost_usd

        # ── Component health ──────────────────────────────────────
        comp_list = self.query_one("#component-list", Vertical)
        await comp_list.remove_children()
        for name, status in health.components.items():
            await comp_list.mount(StatusIndicator(name, status))

        # ── Activity feed ─────────────────────────────────────────
        log = self.query_one("#activity-log", RichLog)
        log.clear()
        if not traces:
            log.write("[dim italic]  📭  No recent activity — run a query to get started[/]")
        else:
            for t in traces:
                root = t.get("root_step", {})
                name = root.get("name", "request") if root else "request"
                dur = t.get("duration_ms", 0)
                icon_map = {"retrieve": "🔍", "ingest": "📥"}
                icon = next((v for k, v in icon_map.items() if k in name), "⚡")
                ts = t.get("start_time", "")[:19]
                err = root.get("error") if root else None
                if err:
                    status = "[#f43f5e]❌[/]"
                elif dur > 1000:
                    status = "[#fbbf24]⚠[/]"
                else:
                    status = "[#34d399]✓[/]"
                log.write(f"{status} {icon} {name:<20} {dur:>7.1f}ms   {ts}")

        # ── Warnings ──────────────────────────────────────────────
        warn_panel = self.query_one("#warnings-panel", Vertical)
        await warn_panel.remove_children()
        if health.warnings:
            await warn_panel.mount(Static("[bold #fbbf24]⚠ Warnings[/]"))
            for w in health.warnings:
                await warn_panel.mount(Static(f"  [#fbbf24]•[/] {w}"))
        elif health.status == "healthy":
            await warn_panel.mount(Static("[#34d399]✓ All systems operational[/]"))
        else:
            await warn_panel.mount(Static(f"[#fbbf24]Status: {health.status}[/]"))
