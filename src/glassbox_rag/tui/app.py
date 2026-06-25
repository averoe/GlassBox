"""
GlassBox RAG — Main TUI Application.

A production-grade terminal dashboard built with Textual.
Replaces the web dashboard entirely.

Launch with:  glassbox-rag tui [--config path] [--refresh 5]
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, TabbedContent, TabPane, Static

from glassbox_rag import __version__
from glassbox_rag.tui.theme import APP_CSS
from glassbox_rag.tui.client import EngineClient
from glassbox_rag.tui.screens.overview import OverviewScreen
from glassbox_rag.tui.screens.pipeline import PipelineScreen
from glassbox_rag.tui.screens.debugger import DebuggerScreen
from glassbox_rag.tui.screens.telemetry import TelemetryScreen
from glassbox_rag.tui.screens.config_editor import ConfigPluginsScreen
from glassbox_rag.tui.screens.ingest import IngestScreen
from glassbox_rag.tui.screens.evaluation import EvaluationScreen
from glassbox_rag.tui.screens.experiments import ExperimentsScreen


# ── Minimum Terminal Size ──────────────────────────────────────────
MIN_COLS = 60
MIN_ROWS = 20


class BrandedHeader(Static):
    """Custom branded header replacing Textual's default."""

    DEFAULT_CSS = """
    BrandedHeader {
        dock: top;
        width: 100%;
        height: 3;
        background: #12141c;
        color: #c4b5fd;
        content-align: center middle;
        text-style: bold;
        border-bottom: solid #2a2e42;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            "⬡  [bold #c4b5fd]GlassBox RAG[/]",
            markup=True,
            **kwargs,
        )


class GlassBoxTUI(App):
    """GlassBox RAG Terminal Dashboard."""

    TITLE = "GlassBox RAG"
    SUB_TITLE = f"v{__version__}"
    CSS = APP_CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("1", "tab_overview", "Overview"),
        Binding("2", "tab_pipeline", "Pipeline"),
        Binding("3", "tab_debugger", "Debugger"),
        Binding("4", "tab_telemetry", "Telemetry"),
        Binding("5", "tab_config", "Config"),
        Binding("6", "tab_ingest", "Ingest"),
        Binding("7", "tab_evaluation", "Eval"),
        Binding("8", "tab_experiments", "Experiments"),
    ]

    def __init__(
        self,
        config_path: str | None = None,
        refresh_interval: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.client = EngineClient(config_path=config_path)
        self.refresh_interval = refresh_interval
        self._timer = None

    def compose(self) -> ComposeResult:
        yield BrandedHeader(id="branded-header")

        with TabbedContent(initial="tab-overview"):
            with TabPane("⬡ Overview", id="tab-overview"):
                yield OverviewScreen(id="overview-screen")

            with TabPane("⛓ Pipeline", id="tab-pipeline"):
                yield PipelineScreen(id="pipeline-screen")

            with TabPane("🔬 Debugger", id="tab-debugger"):
                yield DebuggerScreen(id="debugger-screen")

            with TabPane("📊 Telemetry", id="tab-telemetry"):
                yield TelemetryScreen(id="telemetry-screen")

            with TabPane("⚙ Config & Plugins", id="tab-config"):
                yield ConfigPluginsScreen(id="config-screen")

            with TabPane("📥 Ingest", id="tab-ingest"):
                yield IngestScreen(id="ingest-screen")

            with TabPane("📈 Evaluation", id="tab-evaluation"):
                yield EvaluationScreen(id="evaluation-screen")

            with TabPane("🧪 Experiments", id="tab-experiments"):
                yield ExperimentsScreen(id="experiments-screen")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialise engine and start refresh timer."""
        # ── Terminal size guard ────────────────────────────────────
        size = self.size
        if size.width < MIN_COLS or size.height < MIN_ROWS:
            self.notify(
                f"Terminal is {size.width}×{size.height} — "
                f"GlassBox works best at {MIN_COLS}×{MIN_ROWS} or larger. "
                f"Resize your terminal for the best experience.",
                title="⚠ Small Terminal",
                severity="warning",
                timeout=8,
            )

        self.notify("Initialising GlassBox engine…", title="Starting")

        ok = await self.client.initialize()
        if ok:
            self.notify("Engine ready ✓", title="GlassBox", severity="information")
        else:
            self.notify(
                f"Engine init failed: {self.client.error}",
                title="Warning",
                severity="warning",
            )

        # First data load
        await self._refresh_all()

        # Start periodic refresh
        self._timer = self.set_interval(
            self.refresh_interval,
            self._refresh_all,
        )

    async def _refresh_all(self) -> None:
        """Refresh all visible screens."""
        try:
            overview = self.query_one("#overview-screen", OverviewScreen)
            await overview.refresh_data(self.client)
        except Exception:
            pass

        try:
            debugger = self.query_one("#debugger-screen", DebuggerScreen)
            await debugger.refresh_data(self.client)
        except Exception:
            pass

        try:
            telemetry = self.query_one("#telemetry-screen", TelemetryScreen)
            await telemetry.refresh_data(self.client)
        except Exception:
            pass

        try:
            evaluation = self.query_one("#evaluation-screen", EvaluationScreen)
            await evaluation.refresh_data(self.client)
        except Exception:
            pass

        try:
            experiments = self.query_one("#experiments-screen", ExperimentsScreen)
            await experiments.refresh_data(self.client)
        except Exception:
            pass

    # ── Actions ───────────────────────────────────────────────────

    async def action_refresh(self) -> None:
        """Manual refresh (R key)."""
        self.notify("Refreshing…", title="Refresh")
        await self._refresh_all()

    def action_tab_overview(self) -> None:
        self.query_one(TabbedContent).active = "tab-overview"

    def action_tab_pipeline(self) -> None:
        self.query_one(TabbedContent).active = "tab-pipeline"

    def action_tab_debugger(self) -> None:
        self.query_one(TabbedContent).active = "tab-debugger"

    def action_tab_telemetry(self) -> None:
        self.query_one(TabbedContent).active = "tab-telemetry"

    def action_tab_config(self) -> None:
        self.query_one(TabbedContent).active = "tab-config"

    def action_tab_ingest(self) -> None:
        self.query_one(TabbedContent).active = "tab-ingest"

    def action_tab_evaluation(self) -> None:
        self.query_one(TabbedContent).active = "tab-evaluation"

    def action_tab_experiments(self) -> None:
        self.query_one(TabbedContent).active = "tab-experiments"

    async def action_quit(self) -> None:
        """Shut down engine and exit."""
        if self._timer:
            self._timer.stop()
        await self.client.shutdown()
        self.exit()


def run_tui(config_path: str | None = None, refresh_interval: int = 5) -> None:
    """Entry point for running the TUI."""
    app = GlassBoxTUI(config_path=config_path, refresh_interval=refresh_interval)
    app.run()
