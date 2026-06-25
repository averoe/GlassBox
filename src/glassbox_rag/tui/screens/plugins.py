"""
Plugins Screen — plugin docs, config editor, and API reference.

Features:
  • Plugin type cards (Encoder, Vector Store, Database, Multimodal)
  • Syntax-highlighted documentation viewer
  • Drag-and-drop config ordering
  • Supabase integration guide
  • Inline API reference (replaces /docs)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, RichLog, Markdown

from glassbox_rag.tui.widgets.drag_list import DragList


_PLUGIN_TYPES = [
    ("encoder", "🧠 Encoders", "Text embedding providers (OpenAI, Cohere, HuggingFace)"),
    ("vectorstore", "💾 Vector Stores", "Vector similarity search (Qdrant, Pinecone, Weaviate)"),
    ("database", "🗄️ Databases", "Document storage (PostgreSQL, MongoDB, SQLite)"),
    ("multimodal", "📄 Multimodal", "Content processors (PDF, Image, PPTX extractors)"),
]

_PLUGIN_DOCS = {
    "encoder": """
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
    "vectorstore": """
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
    "database": """
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
    "multimodal": """
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

_API_REFERENCE = """
# GlassBox RAG — API Reference

## Core Engine Methods

### `engine.retrieve(query, top_k=5)`
Execute a retrieval query against the pipeline.

**Parameters:**
| Param   | Type   | Default | Description                          |
|---------|--------|---------|--------------------------------------|
| query   | str    | —       | The search query text                |
| top_k   | int    | 5       | Number of results to return          |

**Returns:** `RetrieveResult` with `.documents`, `.strategy`, `.execution_time_ms`, `.trace_id`

---

### `engine.ingest(documents)`
Ingest documents into the pipeline.

**Parameters:**
| Param     | Type       | Description                          |
|-----------|------------|--------------------------------------|
| documents | list[dict] | Documents with `content` and `metadata` |

---

### `engine.get_metrics_summary()`
Returns a dict with total requests, tokens, cost, and per-operation histograms.

---

### `engine.get_telemetry_status()`
Returns OTel and Prometheus configuration and connection status.

---

### `engine.list_traces(limit=50)`
List recent execution traces.

---

### `engine.get_trace(trace_id)`
Get full trace details including step hierarchy.

---

### `engine.get_cache_stats()`
Returns embedding and query cache hit/miss statistics.

---

### `engine.get_chunk_report()`
Returns chunking statistics and distribution.

---

### `engine.get_hook_info()`
Returns registered lifecycle hooks and their configuration.

---

## Configuration

GlassBox RAG uses YAML configuration. See `config/default.yaml` for full schema.

### Key Sections
- `encoding` — Encoder configuration (model, dimensions, batch size)
- `retrieval` — Retrieval strategy (adaptive, mmr, semantic)
- `vector_store` — Vector store backend (qdrant, pinecone, memory)
- `database` — Document store backend (postgres, mongodb, sqlite)
- `telemetry` — OTel and Prometheus settings
- `security` — Rate limiting, API keys
- `writeback` — Write-back protection (read-only, protected, full)

### Supabase Integration

```yaml
database:
  postgresql:
    host: your-project.supabase.co
    port: 5432
    database: postgres
    user: postgres
    password: your-password
```

Supabase provides PostgreSQL with real-time subscriptions,
authentication, and edge functions that complement GlassBox's capabilities.
"""


class PluginsScreen(Container):
    """Plugin documentation, config editor, and API reference."""

    DEFAULT_CSS = """
    PluginsScreen {
        height: 100%;
        padding: 1 0;
    }
    #plugins-title {
        color: #6366f1;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #plugins-sub {
        color: #8b8fa3;
        padding: 0 0 1 0;
    }
    #plugin-cards {
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
    #config-section {
        margin-top: 1;
        height: auto;
    }
    #config-label {
        color: #8b5cf6;
        text-style: bold;
        margin-bottom: 1;
    }
    #config-hint {
        color: #5c5f73;
        text-style: italic;
        margin-top: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_doc = "api"

    def compose(self) -> ComposeResult:
        yield Static("⬡  Plugin Development & API Reference", id="plugins-title")
        yield Static("Extend GlassBox RAG with custom plugins", id="plugins-sub")

        with Horizontal(id="plugin-cards"):
            for pid, name, desc in _PLUGIN_TYPES:
                with Vertical(classes="plugin-card", id=f"card-{pid}"):
                    yield Static(name, classes="plugin-card-title")
                    yield Static(desc, classes="plugin-card-desc")
                    yield Button(f"View Docs", id=f"btn-{pid}", classes="doc-btn -secondary")

        with Horizontal(id="doc-controls"):
            yield Button("📚 API Reference", id="btn-api", classes="doc-btn-active")
            yield Button("🧠 Encoder", id="btn-doc-encoder", classes="doc-btn -secondary")
            yield Button("💾 Vector Store", id="btn-doc-vectorstore", classes="doc-btn -secondary")
            yield Button("🗄️ Database", id="btn-doc-database", classes="doc-btn -secondary")
            yield Button("📄 Multimodal", id="btn-doc-multimodal", classes="doc-btn -secondary")

        yield Markdown(_API_REFERENCE, id="doc-viewer")

        with Vertical(id="config-section"):
            yield Static("Active Plugin Order  (Alt+↑/↓ to reorder)", id="config-label")
            yield DragList(
                items=[
                    ("enc", "Encoder (sentence-transformers)", "🧠"),
                    ("vec", "Vector Store (qdrant)", "💾"),
                    ("db", "Database (postgresql)", "🗄️"),
                    ("mm", "Multimodal (pdf-extractor)", "📄"),
                ],
                id="plugin-drag-list",
            )
            yield Static("Reorder plugins to change processing priority", id="config-hint")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        viewer = self.query_one("#doc-viewer", Markdown)

        # Plugin card buttons
        if bid.startswith("btn-") and bid.replace("btn-", "") in _PLUGIN_DOCS:
            doc_key = bid.replace("btn-", "")
            viewer.update(_PLUGIN_DOCS[doc_key])
            self._active_doc = doc_key
            self._update_active_btn(f"btn-doc-{doc_key}")
            return

        # Doc control buttons
        if bid == "btn-api":
            viewer.update(_API_REFERENCE)
            self._active_doc = "api"
            self._update_active_btn("btn-api")
        elif bid.startswith("btn-doc-"):
            key = bid.replace("btn-doc-", "")
            if key in _PLUGIN_DOCS:
                viewer.update(_PLUGIN_DOCS[key])
                self._active_doc = key
                self._update_active_btn(bid)

    def _update_active_btn(self, active_id: str) -> None:
        for btn in self.query("#doc-controls Button"):
            if btn.id == active_id:
                btn.set_classes("doc-btn-active")
            else:
                btn.set_classes("doc-btn -secondary")

    async def refresh_data(self, client) -> None:
        """Plugins screen is mostly static — nothing to refresh."""
        pass
