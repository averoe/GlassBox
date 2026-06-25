"""
Evaluation Screen — view evaluation reports and metric trends.

Shows evaluation run history, per-metric aggregate scores,
per-query breakdowns, and regression comparison between runs.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, DataTable, RichLog


class EvaluationScreen(Container):
    """Evaluation reports and metrics dashboard."""

    DEFAULT_CSS = """
    EvaluationScreen {
        height: 100%;
        padding: 1 0;
    }
    #eval-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #eval-subtitle {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #eval-summary-row {
        layout: horizontal;
        height: auto;
        min-height: 6;
        margin: 0 0 1 0;
    }
    .eval-stat-card {
        width: 1fr;
        height: auto;
        min-height: 5;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1;
    }
    .eval-stat-card:hover {
        border: round #6366f1;
    }
    #eval-reports-panel {
        height: 1fr;
        min-height: 12;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    #eval-reports-panel:hover {
        border: round #6366f1;
    }
    #eval-detail-panel {
        height: 1fr;
        min-height: 10;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    #eval-detail-panel:hover {
        border: round #6366f1;
    }
    .panel-title {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("📊  Evaluation Dashboard", id="eval-title")
        yield Static(
            "View evaluation reports, metric scores, and regression analysis",
            id="eval-subtitle",
        )

        with Horizontal(id="eval-summary-row"):
            yield Vertical(
                Static("[bold #8b8fa3]Total Runs[/]"),
                Static("0", id="eval-total-runs"),
                classes="eval-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Avg Faithfulness[/]"),
                Static("—", id="eval-avg-faith"),
                classes="eval-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Avg Groundedness[/]"),
                Static("—", id="eval-avg-ground"),
                classes="eval-stat-card",
            )
            yield Vertical(
                Static("[bold #8b8fa3]Last Run[/]"),
                Static("—", id="eval-last-run"),
                classes="eval-stat-card",
            )

        with Vertical(id="eval-reports-panel"):
            yield Static("Evaluation Reports", classes="panel-title")
            yield DataTable(id="eval-reports-table")

        with Vertical(id="eval-detail-panel"):
            yield Static("Report Detail", classes="panel-title")
            yield RichLog(
                id="eval-detail-log",
                highlight=True,
                markup=True,
                max_lines=100,
            )

    async def on_mount(self) -> None:
        table = self.query_one("#eval-reports-table", DataTable)
        table.add_columns(
            "Timestamp", "Pipeline", "Queries", "Metrics", "Avg Score",
        )
        table.cursor_type = "row"

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show detail when a report row is selected."""
        table = self.query_one("#eval-reports-table", DataTable)
        log = self.query_one("#eval-detail-log", RichLog)
        log.clear()

        try:
            row_data = table.get_row(event.row_key)
            timestamp = row_data[0]

            reports = getattr(self, "_reports", [])
            report = next(
                (r for r in reports if r.get("timestamp", "")[:19] == str(timestamp)[:19]),
                None,
            )
            if not report:
                log.write("[dim italic]  No detail available[/]")
                return

            log.write(f"[bold #c4b5fd]Report: {report.get('timestamp', '')}[/]")
            log.write(f"Pipeline Version: v{report.get('pipeline_version', 0)}")
            log.write(f"Dataset: {report.get('dataset_name', 'N/A')}")
            log.write("")

            agg = report.get("aggregate_scores", {})
            if agg:
                log.write("[bold #8b5cf6]Aggregate Scores:[/]")
                for metric, score in sorted(agg.items()):
                    bar_len = int(score * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    color = "#34d399" if score >= 0.7 else "#fbbf24" if score >= 0.4 else "#f43f5e"
                    log.write(f"  [{color}]{bar}[/] {metric}: {score:.4f}")
                log.write("")

            per_query = report.get("per_query_scores", [])
            if per_query:
                log.write(f"[bold #8b5cf6]Per-Query Breakdown ({len(per_query)} queries):[/]")
                for qs in per_query[:10]:
                    query_preview = qs.get("query", "")[:60]
                    scores = qs.get("scores", {})
                    scores_str = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
                    log.write(f"  [#22d3ee]▸[/] '{query_preview}' → {scores_str}")
                if len(per_query) > 10:
                    log.write(f"  [dim]… and {len(per_query) - 10} more[/]")
        except Exception as exc:
            log.write(f"[#f43f5e]Error loading detail: {exc}[/]")

    async def refresh_data(self, client) -> None:
        """Pull evaluation data from the client."""
        reports = await client.get_evaluations(limit=20)
        self._reports = reports

        # Update summary cards
        try:
            self.query_one("#eval-total-runs", Static).update(
                f"[bold #e8eaf0]{len(reports)}[/]"
            )
        except Exception:
            pass

        if reports:
            latest = reports[0]
            agg = latest.get("aggregate_scores", {})

            faith = agg.get("faithfulness", None)
            ground = agg.get("groundedness", None)

            try:
                self.query_one("#eval-avg-faith", Static).update(
                    f"[bold #e8eaf0]{faith:.4f}[/]" if faith is not None else "—"
                )
            except Exception:
                pass

            try:
                self.query_one("#eval-avg-ground", Static).update(
                    f"[bold #e8eaf0]{ground:.4f}[/]" if ground is not None else "—"
                )
            except Exception:
                pass

            try:
                ts = latest.get("timestamp", "—")[:19]
                self.query_one("#eval-last-run", Static).update(
                    f"[bold #e8eaf0]{ts}[/]"
                )
            except Exception:
                pass

        # Update reports table
        table = self.query_one("#eval-reports-table", DataTable)
        table.clear()
        for report in reports:
            ts = report.get("timestamp", "")[:19]
            pv = f"v{report.get('pipeline_version', 0)}"
            n_queries = len(report.get("per_query_scores", []))
            metrics = ", ".join(report.get("metrics_requested", [])[:3])
            agg = report.get("aggregate_scores", {})
            if agg:
                avg = sum(agg.values()) / len(agg)
                avg_str = f"{avg:.4f}"
            else:
                avg_str = "—"
            table.add_row(ts, pv, str(n_queries), metrics, avg_str)

        if not reports:
            log = self.query_one("#eval-detail-log", RichLog)
            log.clear()
            log.write(
                "[dim italic]  📭  No evaluation reports found.\n"
                "  Run: glassbox-rag eval run --dataset <path>[/]"
            )
