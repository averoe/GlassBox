"""
Experiments Screen — view and compare pipeline experiments.

Shows experiment history with pipeline versions, git commits,
evaluation results, and side-by-side comparison of two experiments.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, DataTable, RichLog


class ExperimentsScreen(Container):
    """Experiment tracking and comparison dashboard."""

    DEFAULT_CSS = """
    ExperimentsScreen {
        height: 100%;
        padding: 1 0;
    }
    #exp-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #exp-subtitle {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #exp-summary-row {
        layout: horizontal;
        height: auto;
        min-height: 6;
        margin: 0 0 1 0;
    }
    .exp-stat-card {
        width: 1fr;
        height: auto;
        min-height: 5;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1;
    }
    .exp-stat-card:hover {
        border: round #6366f1;
    }
    #exp-table-panel {
        height: 1fr;
        min-height: 12;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    #exp-table-panel:hover {
        border: round #6366f1;
    }
    #exp-detail-panel {
        height: 1fr;
        min-height: 10;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    #exp-detail-panel:hover {
        border: round #6366f1;
    }
    .panel-title {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🧪  Experiments Dashboard", id="exp-title")
        yield Static(
            "Track pipeline runs, compare versions, and monitor regressions",
            id="exp-subtitle",
        )

        with Horizontal(id="exp-summary-row"):
            yield Vertical(
                Static("[bold #8b8fa3]Total Experiments[/]"),
                Static("0", id="exp-total"),
                classes="exp-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Latest Version[/]"),
                Static("—", id="exp-latest-ver"),
                classes="exp-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Git Commit[/]"),
                Static("—", id="exp-git-commit"),
                classes="exp-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Last Experiment[/]"),
                Static("—", id="exp-last-ts"),
                classes="exp-stat-card",
            )

        with Vertical(id="exp-table-panel"):
            yield Static("Experiment History", classes="panel-title")
            yield DataTable(id="exp-table")

        with Vertical(id="exp-detail-panel"):
            yield Static("Experiment Detail", classes="panel-title")
            yield RichLog(
                id="exp-detail-log",
                highlight=True,
                markup=True,
                max_lines=100,
            )

    async def on_mount(self) -> None:
        table = self.query_one("#exp-table", DataTable)
        table.add_columns(
            "ID", "Timestamp", "Pipeline", "Git", "Dataset", "Metrics",
        )
        table.cursor_type = "row"

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show experiment detail when a row is selected."""
        table = self.query_one("#exp-table", DataTable)
        log = self.query_one("#exp-detail-log", RichLog)
        log.clear()

        try:
            row_data = table.get_row(event.row_key)
            exp_id = str(row_data[0])

            experiments = getattr(self, "_experiments", [])
            exp = next(
                (e for e in experiments if e.get("experiment_id") == exp_id),
                None,
            )
            if not exp:
                log.write("[dim italic]  No detail available[/]")
                return

            log.write(f"[bold #c4b5fd]Experiment: {exp.get('experiment_id', '')}[/]")
            log.write(f"Timestamp:        {exp.get('timestamp', '')}")
            log.write(f"Pipeline Version: v{exp.get('pipeline_version', 0)}")
            log.write(f"Git Commit:       {exp.get('git_commit', 'N/A')}")
            log.write(f"Dataset:          {exp.get('dataset_version', 'N/A')}")
            log.write(f"Trace ID:         {exp.get('trace_id', 'N/A')[:16]}")
            log.write(f"Snapshot:         {exp.get('snapshot_id', 'N/A')}")
            log.write("")

            eval_results = exp.get("evaluation_results", {})
            if eval_results:
                log.write("[bold #8b5cf6]Evaluation Results:[/]")
                for metric, score in sorted(eval_results.items()):
                    if isinstance(score, (int, float)):
                        bar_len = int(float(score) * 20)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        color = (
                            "#34d399" if score >= 0.7
                            else "#fbbf24" if score >= 0.4
                            else "#f43f5e"
                        )
                        log.write(f"  [{color}]{bar}[/] {metric}: {score:.4f}")
                    else:
                        log.write(f"  {metric}: {score}")
            else:
                log.write("[dim italic]  No evaluation results recorded[/]")

            metadata = exp.get("metadata", {})
            if metadata:
                log.write("")
                log.write("[bold #8b5cf6]Metadata:[/]")
                for k, v in metadata.items():
                    log.write(f"  {k}: {v}")

        except Exception as exc:
            log.write(f"[#f43f5e]Error loading detail: {exc}[/]")

    async def refresh_data(self, client) -> None:
        """Pull experiment data from the client."""
        experiments = await client.get_experiments(limit=20)
        self._experiments = experiments

        # Update summary cards
        try:
            self.query_one("#exp-total", Static).update(
                f"[bold #e8eaf0]{len(experiments)}[/]"
            )
        except Exception:
            pass

        if experiments:
            latest = experiments[0]

            try:
                self.query_one("#exp-latest-ver", Static).update(
                    f"[bold #e8eaf0]v{latest.get('pipeline_version', 0)}[/]"
                )
            except Exception:
                pass

            try:
                gc = latest.get("git_commit", "—") or "—"
                self.query_one("#exp-git-commit", Static).update(
                    f"[bold #e8eaf0]{gc}[/]"
                )
            except Exception:
                pass

            try:
                ts = latest.get("timestamp", "—")[:19]
                self.query_one("#exp-last-ts", Static).update(
                    f"[bold #e8eaf0]{ts}[/]"
                )
            except Exception:
                pass

        # Update table
        table = self.query_one("#exp-table", DataTable)
        table.clear()
        for exp in experiments:
            exp_id = exp.get("experiment_id", "")
            ts = exp.get("timestamp", "")[:19]
            pv = f"v{exp.get('pipeline_version', 0)}"
            git = exp.get("git_commit", "—") or "—"
            dataset = exp.get("dataset_version", "—") or "—"
            eval_res = exp.get("evaluation_results", {})
            if eval_res:
                metrics_summary = ", ".join(
                    f"{k}={v:.2f}" for k, v in list(eval_res.items())[:3]
                    if isinstance(v, (int, float))
                )
            else:
                metrics_summary = "—"
            table.add_row(exp_id, ts, pv, git, dataset, metrics_summary)

        if not experiments:
            log = self.query_one("#exp-detail-log", RichLog)
            log.clear()
            log.write(
                "[dim italic]  📭  No experiments recorded yet.\n"
                "  Run: glassbox-rag experiment record[/]"
            )
