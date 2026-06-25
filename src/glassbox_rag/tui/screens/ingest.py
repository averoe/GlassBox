"""
Ingest Screen — document ingestion panel for the TUI.

Features:
  • Text paste area for raw content
  • File path input for file-based ingestion
  • Metadata key/value editor
  • Ingest progress display
  • Recent ingestions table
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, Input, TextArea, DataTable, RichLog

from glassbox_rag.tui.widgets.collapsible_panel import CollapsiblePanel


class IngestScreen(Container):
    """Document ingestion panel."""

    DEFAULT_CSS = """
    IngestScreen {
        height: 100%;
        padding: 1 0;
    }
    #ingest-title {
        color: #c4b5fd;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #ingest-sub {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #ingest-top {
        layout: horizontal;
        height: auto;
        min-height: 14;
        margin-bottom: 1;
    }
    #text-input-panel {
        width: 2fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin-right: 1;
    }
    #text-input-panel:hover {
        border: round #6366f1;
    }
    #file-meta-panel {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
    }
    #file-meta-panel:hover {
        border: round #6366f1;
    }
    .section-label {
        color: #c4b5fd;
        text-style: bold;
        margin-bottom: 1;
    }
    #ingest-text {
        min-height: 8;
        max-height: 12;
    }
    #file-path-input {
        margin-bottom: 1;
    }
    #meta-key, #meta-value {
        width: 1fr;
        margin-right: 1;
    }
    #meta-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #add-meta-btn {
        width: auto;
        min-width: 8;
        background: #22263a;
        border: tall #2a2e42;
    }
    #ingest-controls {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    #ingest-btn {
        width: auto;
        min-width: 22;
        margin-right: 1;
    }
    #clear-btn {
        width: auto;
        min-width: 14;
        background: #22263a;
        border: tall #2a2e42;
    }
    #ingest-output {
        min-height: 4;
        max-height: 8;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin-bottom: 1;
    }
    #recent-ingests {
        height: 1fr;
        min-height: 6;
    }
    #meta-tags {
        height: auto;
        max-height: 4;
        color: #8b8fa3;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metadata: dict[str, str] = {}
        self._recent: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("⬡  Document Ingestion", id="ingest-title")
        yield Static("Ingest text, files, or paste content into the pipeline", id="ingest-sub")

        with Horizontal(id="ingest-top"):
            with Vertical(id="text-input-panel"):
                yield Static("📝  Paste Content", classes="section-label")
                yield TextArea(
                    "",
                    language="markdown",
                    id="ingest-text",
                )

            with Vertical(id="file-meta-panel"):
                yield Static("📁  File Path", classes="section-label")
                yield Input(
                    placeholder="path/to/document.txt or directory/",
                    id="file-path-input",
                )
                yield Static("🏷  Metadata Tags", classes="section-label")
                with Horizontal(id="meta-row"):
                    yield Input(placeholder="key", id="meta-key")
                    yield Input(placeholder="value", id="meta-value")
                    yield Button("+ Add", id="add-meta-btn", classes="-secondary")
                yield Static("", id="meta-tags")

        with Horizontal(id="ingest-controls"):
            yield Button("📥  Ingest Documents", id="ingest-btn", variant="primary")
            yield Button("🗑  Clear", id="clear-btn", classes="-secondary")

        yield RichLog(id="ingest-output", highlight=True, markup=True, max_lines=50)

        with CollapsiblePanel(title="Recent Ingestions", collapsed=False):
            yield DataTable(id="recent-ingests", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#recent-ingests", DataTable)
        table.add_column("Time", width=20, key="time")
        table.add_column("Source", width=24, key="source")
        table.add_column("Chunks", width=10, key="chunks")
        table.add_column("Vectors", width=10, key="vectors")
        table.add_column("Status", width=10, key="status")

        log = self.query_one("#ingest-output", RichLog)
        log.write("[#5c5f73]  📭  Ready to ingest — paste text or specify a file path[/]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid == "add-meta-btn":
            self._add_metadata_tag()
        elif bid == "ingest-btn":
            await self._run_ingest()
        elif bid == "clear-btn":
            self._clear_form()

    def _add_metadata_tag(self) -> None:
        try:
            key_input = self.query_one("#meta-key", Input)
            val_input = self.query_one("#meta-value", Input)
            key = key_input.value.strip()
            val = val_input.value.strip()
            if key:
                self._metadata[key] = val
                key_input.value = ""
                val_input.value = ""
                self._refresh_meta_display()
        except Exception:
            pass

    def _refresh_meta_display(self) -> None:
        tags_display = self.query_one("#meta-tags", Static)
        if self._metadata:
            parts = [f"[#c4b5fd]{k}[/]=[#8b8fa3]{v}[/]" for k, v in self._metadata.items()]
            tags_display.update("  ".join(parts))
        else:
            tags_display.update("")

    def _clear_form(self) -> None:
        try:
            self.query_one("#ingest-text", TextArea).clear()
            self.query_one("#file-path-input", Input).value = ""
            self.query_one("#meta-key", Input).value = ""
            self.query_one("#meta-value", Input).value = ""
            self._metadata.clear()
            self._refresh_meta_display()
            log = self.query_one("#ingest-output", RichLog)
            log.clear()
            log.write("[#5c5f73]  📭  Form cleared[/]")
        except Exception:
            pass

    async def _run_ingest(self) -> None:
        log = self.query_one("#ingest-output", RichLog)
        log.clear()

        app = self.app
        client = getattr(app, "client", None)
        if not client or not client.ready:
            log.write("[red]✗ Engine not ready — cannot ingest[/]")
            return

        # Gather documents
        documents: list[dict] = []
        text_area = self.query_one("#ingest-text", TextArea)
        file_input = self.query_one("#file-path-input", Input)

        pasted_text = text_area.text.strip()
        file_path_str = file_input.value.strip()

        if pasted_text:
            documents.append({
                "content": pasted_text,
                "metadata": {**self._metadata, "source": "tui_paste"},
            })
            log.write(f"[#c4b5fd]📝 Pasted text:[/] {len(pasted_text)} characters")

        if file_path_str:
            file_path = Path(file_path_str)
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    documents.append({
                        "content": content,
                        "metadata": {
                            **self._metadata,
                            "source": str(file_path),
                            "filename": file_path.name,
                        },
                    })
                    log.write(f"[#c4b5fd]📁 File:[/] {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    log.write(f"[red]✗ Could not read file: {e}[/]")
                    return
            elif file_path.is_dir():
                file_count = 0
                for fp in file_path.rglob("*"):
                    if fp.is_file() and fp.suffix in (".txt", ".md", ".csv", ".json"):
                        try:
                            content = fp.read_text(encoding="utf-8", errors="replace")
                            documents.append({
                                "content": content,
                                "metadata": {
                                    **self._metadata,
                                    "source": str(fp),
                                    "filename": fp.name,
                                },
                            })
                            file_count += 1
                        except Exception:
                            pass
                log.write(f"[#c4b5fd]📁 Directory:[/] {file_count} files from {file_path}")
            else:
                log.write(f"[red]✗ Path not found: {file_path_str}[/]")
                return

        if not documents:
            log.write("[yellow]⚠ Nothing to ingest — paste text or specify a file path[/]")
            return

        log.write(f"[#8b8fa3]Ingesting {len(documents)} document(s)…[/]")

        try:
            result = await client.run_ingest(documents)
            if "error" in result:
                log.write(f"[red]✗ Ingest failed: {result['error']}[/]")
            else:
                chunks = result.get("chunks_created", 0)
                doc_ids = result.get("document_ids", [])
                trace_id = result.get("trace_id", "")
                log.write(f"[#34d399]✓ Ingested successfully![/]")
                log.write(f"  Documents: {result.get('documents_ingested', len(documents))}")
                log.write(f"  Chunks:    {chunks}")
                log.write(f"  Vectors:   {len(doc_ids)}")
                if trace_id:
                    log.write(f"  Trace ID:  {trace_id[:12]}…")

                # Add to recent table
                import datetime
                self._recent.insert(0, {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": file_path_str or "paste",
                    "chunks": str(chunks),
                    "vectors": str(len(doc_ids)),
                    "status": "✓",
                })
                self._refresh_recent_table()
        except Exception as e:
            log.write(f"[red]✗ Ingest error: {e}[/]")

    def _refresh_recent_table(self) -> None:
        try:
            table = self.query_one("#recent-ingests", DataTable)
            table.clear()
            for r in self._recent[:20]:
                table.add_row(
                    r["time"], r["source"][:22], r["chunks"], r["vectors"], r["status"],
                )
        except Exception:
            pass

    async def refresh_data(self, client) -> None:
        """Nothing to auto-refresh for ingest."""
        pass
