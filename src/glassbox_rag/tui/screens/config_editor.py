"""
Config & Plugins Screen — YAML config editor + plugin documentation.

Features:
  • Live YAML config viewer with syntax highlighting
  • Editable YAML with save-to-file support
  • Plugin type cards and documentation browser
  • Section navigation
"""

from __future__ import annotations

from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, TextArea, Markdown, Input

from glassbox_rag.tui.widgets.collapsible_panel import CollapsiblePanel


# ── Plugin Docs ──────────────────────────────────────────────────

_PLUGIN_DOCS = {
    "encoder": """\
## Creating a Custom Encoder

Inherit from `BaseEncoder` and implement the `encode` method:

```python
from glassbox_rag.core.encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    async def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.call_my_api(texts)
        return np.array(embeddings)
```

### Configuration

```yaml
encoding:
  local:
    my_encoder:
      type: "my_encoder"
      api_key: "your-key"
```
""",
    "vectorstore": """\
## Creating a Custom Vector Store

Inherit from `VectorStorePlugin`:

```python
from glassbox_rag.plugins.base import VectorStorePlugin

class MyVectorStore(VectorStorePlugin):
    async def add_vectors(self, vectors, metadata):
        pass  # Store vectors

    async def search(self, query_vector, top_k):
        pass  # Return results
```
""",
    "database": """\
## Creating a Custom Database

Inherit from `DatabasePlugin`:

```python
from glassbox_rag.plugins.base import DatabasePlugin

class MyDatabase(DatabasePlugin):
    async def store_document(self, doc_id, content, metadata):
        pass

    async def get_document(self, doc_id):
        pass
```
""",
    "multimodal": """\
## Creating a Multimodal Processor

Inherit from `BaseMultimodalProcessor`:

```python
from glassbox_rag.utils.multimodal import BaseMultimodalProcessor

class MyImageProcessor(BaseMultimodalProcessor):
    async def process(self, content):
        texts = self.ocr_my_image(content.content)
        return texts
```
""",
}

_API_REFERENCE = """\
# GlassBox RAG — API Reference

## Core Engine Methods

### `engine.retrieve(query, top_k=5)`
Execute a retrieval query against the pipeline.

**Returns:** `RetrievalResult` with `.documents`, `.strategy`, `.execution_time_ms`, `.trace_id`

---

### `engine.ingest(documents)`
Ingest documents into the pipeline.

---

### `engine.generate(query)`
Retrieve + Generate: end-to-end RAG pipeline.

---

### `engine.get_metrics_summary()`
Returns a dict with total requests, tokens, cost, and per-operation histograms.

---

### `engine.get_cache_stats()`
Returns embedding cache hit/miss statistics.

---

### `engine.get_hook_info()`
Returns registered lifecycle hooks and their configuration.

---

## Configuration Sections

- `encoding` — Encoder configuration (model, dimensions, batch size)
- `retrieval` — Retrieval strategy (adaptive, mmr, semantic)
- `vector_store` — Vector store backend (qdrant, pinecone, memory)
- `database` — Document store backend (postgres, mongodb, sqlite)
- `telemetry` — OTel and Prometheus settings
- `security` — Rate limiting, API keys
- `writeback` — Write-back protection (read-only, protected, full)
- `generation` — LLM backend configuration
- `chunking` — Text chunking strategy and parameters
"""


class ConfigPluginsScreen(Container):
    """Configuration editor + plugin documentation."""

    DEFAULT_CSS = """
    ConfigPluginsScreen {
        height: 100%;
        padding: 1 0;
    }
    #cp-title {
        color: #c4b5fd;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #cp-sub {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #cp-mode-bar {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    .mode-btn {
        margin-right: 1;
        background: #22263a;
        border: tall #2a2e42;
    }
    .mode-btn-active {
        background: #6366f1;
        border: tall #6366f1;
        margin-right: 1;
    }
    #cp-body {
        height: 1fr;
    }
    #yaml-editor-panel {
        height: 100%;
    }
    #yaml-editor {
        height: 1fr;
        min-height: 16;
    }
    #yaml-controls {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    #save-btn {
        width: auto;
        min-width: 18;
        margin-right: 1;
    }
    #reload-btn {
        width: auto;
        min-width: 18;
        background: #22263a;
        border: tall #2a2e42;
        margin-right: 1;
    }
    #save-path-input {
        width: 1fr;
    }
    #yaml-status {
        color: #8b8fa3;
        margin-top: 1;
        height: auto;
    }
    #plugins-panel {
        height: 100%;
    }
    #plugin-cards-row {
        layout: horizontal;
        height: auto;
        min-height: 5;
        margin-bottom: 1;
    }
    .plugin-card {
        width: 1fr;
        background: #181b27;
        border: round #2a2e42;
        padding: 1 2;
        margin: 0 1 0 0;
    }
    .plugin-card:hover {
        border: round #6366f1;
    }
    .plugin-card-title {
        text-style: bold;
        color: #c4b5fd;
    }
    .plugin-card-desc {
        color: #8b8fa3;
    }
    #doc-controls {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    .doc-btn {
        margin-right: 1;
        background: #22263a;
        border: tall #2a2e42;
    }
    .doc-btn-active {
        background: #6366f1;
        border: tall #6366f1;
        margin-right: 1;
    }
    #doc-viewer {
        height: 1fr;
        background: #181b27;
        border: round #2a2e42;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = "config"  # "config" or "plugins"
        self._config_path: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("⬡  Configuration & Plugins", id="cp-title")
        yield Static("Edit YAML config or browse plugin documentation", id="cp-sub")

        with Horizontal(id="cp-mode-bar"):
            yield Button("⚙  Config Editor", id="mode-config", classes="mode-btn-active")
            yield Button("🔌 Plugins & Docs", id="mode-plugins", classes="mode-btn")

        with Container(id="cp-body"):
            # Config editor (default visible)
            with Vertical(id="yaml-editor-panel"):
                yield TextArea(
                    "# Loading configuration...\n",
                    language="yaml",
                    id="yaml-editor",
                    show_line_numbers=True,
                )
                with Horizontal(id="yaml-controls"):
                    yield Button("💾  Save Config", id="save-btn", variant="primary")
                    yield Button("🔄  Reload", id="reload-btn", classes="-secondary")
                    yield Input(
                        placeholder="Save path (e.g. config/default.yaml)",
                        id="save-path-input",
                    )
                yield Static("", id="yaml-status")

            # Plugins panel (hidden by default)
            with Vertical(id="plugins-panel"):
                with Horizontal(id="plugin-cards-row"):
                    with Vertical(classes="plugin-card"):
                        yield Static("🧠 Encoders", classes="plugin-card-title")
                        yield Static("Text embedding providers", classes="plugin-card-desc")
                        yield Button("View Docs", id="btn-encoder", classes="doc-btn -secondary")
                    with Vertical(classes="plugin-card"):
                        yield Static("💾 Vector Stores", classes="plugin-card-title")
                        yield Static("Vector similarity search", classes="plugin-card-desc")
                        yield Button("View Docs", id="btn-vectorstore", classes="doc-btn -secondary")
                    with Vertical(classes="plugin-card"):
                        yield Static("🗄️ Databases", classes="plugin-card-title")
                        yield Static("Document storage backends", classes="plugin-card-desc")
                        yield Button("View Docs", id="btn-database", classes="doc-btn -secondary")
                    with Vertical(classes="plugin-card"):
                        yield Static("📄 Multimodal", classes="plugin-card-title")
                        yield Static("Content processors", classes="plugin-card-desc")
                        yield Button("View Docs", id="btn-multimodal", classes="doc-btn -secondary")

                with Horizontal(id="doc-controls"):
                    yield Button("📚 API Reference", id="btn-api", classes="doc-btn-active")
                    yield Button("🧠 Encoder", id="btn-doc-encoder", classes="doc-btn")
                    yield Button("💾 Vector Store", id="btn-doc-vectorstore", classes="doc-btn")
                    yield Button("🗄️ Database", id="btn-doc-database", classes="doc-btn")
                    yield Button("📄 Multimodal", id="btn-doc-multimodal", classes="doc-btn")

                yield Markdown(_API_REFERENCE, id="doc-viewer")

    def on_mount(self) -> None:
        # Start with plugins panel hidden
        try:
            self.query_one("#plugins-panel").display = False
        except Exception:
            pass

        # Load config
        self._load_config()

    def _load_config(self) -> None:
        """Load the config from the engine client into the YAML editor."""
        app = self.app
        client = getattr(app, "client", None)
        if not client:
            return

        try:
            config = client.config
            # Serialize the Pydantic model to a dict, then to YAML
            config_dict = config.model_dump(
                exclude_defaults=False,
                exclude_none=True,
            )
            # Remove secrets (replace SecretStr values with placeholder)
            config_dict = self._sanitize_secrets(config_dict)
            yaml_str = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False, indent=2,
            )
            editor = self.query_one("#yaml-editor", TextArea)
            editor.clear()
            editor.insert(yaml_str)

            # Set save path from client config path
            config_path = getattr(client, "_config_path", None)
            if config_path:
                self._config_path = config_path
                try:
                    self.query_one("#save-path-input", Input).value = str(config_path)
                except Exception:
                    pass

            status = self.query_one("#yaml-status", Static)
            status.update("[#34d399]✓ Config loaded from engine[/]")
        except Exception as e:
            status = self.query_one("#yaml-status", Static)
            status.update(f"[red]✗ Could not load config: {e}[/]")

    @staticmethod
    def _sanitize_secrets(obj):
        """Replace SecretStr values with masked placeholders."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if hasattr(v, "get_secret_value"):
                    result[k] = "***"
                else:
                    result[k] = ConfigPluginsScreen._sanitize_secrets(v)
            return result
        elif isinstance(obj, list):
            return [ConfigPluginsScreen._sanitize_secrets(item) for item in obj]
        elif hasattr(obj, "get_secret_value"):
            return "***"
        return obj

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        # Mode switching
        if bid == "mode-config":
            self._switch_mode("config")
            return
        elif bid == "mode-plugins":
            self._switch_mode("plugins")
            return

        # Config actions
        if bid == "save-btn":
            self._save_config()
        elif bid == "reload-btn":
            self._load_config()

        # Plugin card buttons
        if bid.startswith("btn-") and bid.replace("btn-", "") in _PLUGIN_DOCS:
            doc_key = bid.replace("btn-", "")
            self.query_one("#doc-viewer", Markdown).update(_PLUGIN_DOCS[doc_key])
            self._update_doc_btn(f"btn-doc-{doc_key}")
            return

        # Doc control buttons
        if bid == "btn-api":
            self.query_one("#doc-viewer", Markdown).update(_API_REFERENCE)
            self._update_doc_btn("btn-api")
        elif bid.startswith("btn-doc-"):
            key = bid.replace("btn-doc-", "")
            if key in _PLUGIN_DOCS:
                self.query_one("#doc-viewer", Markdown).update(_PLUGIN_DOCS[key])
                self._update_doc_btn(bid)

    def _switch_mode(self, mode: str) -> None:
        self._mode = mode
        try:
            yaml_panel = self.query_one("#yaml-editor-panel")
            plugins_panel = self.query_one("#plugins-panel")
            config_btn = self.query_one("#mode-config", Button)
            plugins_btn = self.query_one("#mode-plugins", Button)

            if mode == "config":
                yaml_panel.display = True
                plugins_panel.display = False
                config_btn.set_classes("mode-btn-active")
                plugins_btn.set_classes("mode-btn")
            else:
                yaml_panel.display = False
                plugins_panel.display = True
                config_btn.set_classes("mode-btn")
                plugins_btn.set_classes("mode-btn-active")
        except Exception:
            pass

    def _save_config(self) -> None:
        """Save the YAML editor content to a file."""
        status = self.query_one("#yaml-status", Static)
        try:
            editor = self.query_one("#yaml-editor", TextArea)
            yaml_text = editor.text

            # Validate YAML
            try:
                parsed = yaml.safe_load(yaml_text)
                if not isinstance(parsed, dict):
                    status.update("[red]✗ Invalid YAML: root must be a mapping[/]")
                    return
            except yaml.YAMLError as e:
                status.update(f"[red]✗ YAML syntax error: {e}[/]")
                return

            # Get save path
            path_input = self.query_one("#save-path-input", Input)
            save_path = path_input.value.strip()
            if not save_path:
                save_path = self._config_path or "config/default.yaml"

            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_text, encoding="utf-8")

            status.update(f"[#34d399]✓ Config saved to {path}[/]")
            self.notify(f"Config saved to {path}", title="Saved")

        except Exception as e:
            status.update(f"[red]✗ Save failed: {e}[/]")

    def _update_doc_btn(self, active_id: str) -> None:
        for btn in self.query("#doc-controls Button"):
            if btn.id == active_id:
                btn.set_classes("doc-btn-active")
            else:
                btn.set_classes("doc-btn")

    async def refresh_data(self, client) -> None:
        """Config screen is mostly static — nothing to auto-refresh."""
        pass
