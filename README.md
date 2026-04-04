# GlassBox

An open-source, high-transparency modular RAG (Retrieval-Augmented Generation) framework for AI/ML applications.

## Features

- Modular architecture with pluggable components
- Support for multiple vector stores (Qdrant, Chroma)
- Multiple embedding providers (OpenAI, Ollama, ONNX)
- Database support (PostgreSQL, SQLite)
- Comprehensive tracing and monitoring
- FastAPI-based REST API
- Web dashboard for trace visualization
- Async/await support with asyncio
- Pydantic v2 for robust configuration management

## Installation

Install from PyPI:

```bash
pip install glassbox-rag
```

## Quick Start

```python
from glassbox_rag.core.engine import GlassBoxEngine
from glassbox_rag.config import GlassBoxConfig

# Initialize the engine
config = GlassBoxConfig()
engine = GlassBoxEngine(config)

# Ingest documents
documents = [
    {"content": "Document 1", "metadata": {"source": "source1"}},
    {"content": "Document 2", "metadata": {"source": "source2"}},
]
engine.ingest(documents)

# Retrieve relevant documents
results = engine.retrieve("query text", top_k=5)
```

## Configuration

Configure via YAML file (config/default.yaml):

```yaml
server:
  host: "0.0.0.0"
  port: 8000

vector_store:
  type: "qdrant"
  config:
    url: "http://localhost:6333"

encoder:
  type: "openai"
  config:
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"

database:
  type: "postgresql"
  config:
    url: "postgresql://user:password@localhost/glassbox"
```

## API Endpoints

- GET /health - Health check
- POST /retrieve - Retrieve documents
- POST /ingest - Ingest new documents
- POST /update - Update existing documents
- GET /traces/{id} - Get execution trace
- GET /traces/{id}/visualize - Visualize trace

## Running the Server

Start the FastAPI server with Uvicorn:

```bash
python -m glassbox_rag
```

The server will be available at http://localhost:8000

## Web Dashboard

Access the web dashboard at http://localhost:8000/ to:

- View recent execution traces
- Visualize trace hierarchy and timing
- Monitor system metrics
- Track token usage and costs

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Plugin Architecture

Extend GlassBox with custom plugins:

1. Create a plugin class inheriting from the appropriate base class
2. Implement required abstract methods
3. Register in the configuration

Example custom embedder:

```python
from glassbox_rag.plugins.base import EmbedderBase

class CustomEmbedder(EmbedderBase):
    def encode(self, texts):
        # Your embedding implementation
        return embeddings
```

## Architecture

GlassBox consists of several core components:

- Engine: Orchestrates all operations
- Encoder: Handles document and query encoding
- Retriever: Implements retrieval strategies
- Writeback: Manages document updates with protection
- Metrics: Tracks token usage and costs
- Trace: Records execution traces for debugging

## License

Licensed under the Apache License 2.0. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.