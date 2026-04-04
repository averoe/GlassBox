"""
Basic retrieval example.

Demonstrates simple document retrieval using GlassBox RAG.
"""

import asyncio
from glassbox_rag import GlassBoxEngine
from glassbox_rag.config import load_config


async def main():
    """Run basic retrieval example."""
    # Load configuration
    config = load_config("config/default.yaml")
    
    # Create engine
    engine = GlassBoxEngine(config)
    
    # First, ingest some sample documents
    print("Ingesting documents...")
    documents = [
        {
            "content": "GlassBox RAG is a transparent modular RAG framework.",
            "metadata": {"source": "docs"},
        },
        {
            "content": "Retrieval Augmented Generation combines retrieval with generation.",
            "metadata": {"source": "docs"},
        },
        {
            "content": "The modular encoding layer supports both local and cloud models.",
            "metadata": {"source": "docs"},
        },
    ]
    
    ingest_result = await engine.ingest(documents)
    print(f"✓ Ingested {ingest_result['documents_ingested']} documents")
    print(f"  Trace ID: {ingest_result['trace_id']}\n")
    
    # Perform a retrieval
    print("Retrieving documents...")
    query = "What is RAG?"
    result, trace = await engine.retrieve(query, top_k=2)
    
    print(f"✓ Retrieved {result.total_results} documents for query: '{query}'")
    print(f"  Strategy used: {result.strategy}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  Trace ID: {result.query}")
    
    # Display trace
    print("\nExecution Trace:")
    print(trace)


if __name__ == "__main__":
    asyncio.run(main())
