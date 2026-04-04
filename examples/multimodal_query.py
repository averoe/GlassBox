"""
Multimodal query example.

Demonstrates retrieval with multimodal content (text, images, PDFs).
"""

import asyncio
from pathlib import Path
from glassbox_rag import GlassBoxEngine
from glassbox_rag.config import load_config
from glassbox_rag.utils.multimodal import MultimodalContent, MultimodalHandler


async def main():
    """Run multimodal example."""
    # Load configuration
    config = load_config("config/default.yaml")
    
    # Check if multimodal is enabled
    if not config.multimodal.enabled:
        print("Multimodal is disabled in config")
        return
    
    # Create engine
    engine = GlassBoxEngine(config)
    
    # Example: Process multimodal content
    handler = MultimodalHandler()
    
    # Text content
    text_content = MultimodalContent(
        content_type="text",
        content="GlassBox RAG supports unified multimodal retrieval.",
    )
    
    print("Processing text content...")
    text_chunks = await handler.process(text_content)
    print(f"✓ Extracted {len(text_chunks)} text chunks")
    
    # Image content (placeholder)
    image_content = MultimodalContent(
        content_type="image",
        content=b"fake image data",  # In real scenario, this would be actual image bytes
    )
    
    print("\nProcessing image content...")
    try:
        image_chunks = await handler.process(image_content)
        print(f"✓ Extracted {len(image_chunks)} chunks from image")
    except Exception as e:
        print(f"  Note: Image processing not fully implemented ({e})")
    
    # PDF content (placeholder)
    pdf_content = MultimodalContent(
        content_type="pdf",
        content=b"fake pdf data",  # In real scenario, this would be actual PDF bytes
    )
    
    print("\nProcessing PDF content...")
    try:
        pdf_chunks = await handler.process(pdf_content)
        print(f"✓ Extracted {len(pdf_chunks)} chunks from PDF")
    except Exception as e:
        print(f"  Note: PDF processing not fully implemented ({e})")


if __name__ == "__main__":
    asyncio.run(main())
