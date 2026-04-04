"""
Write-back example.

Demonstrates safe bi-directional write operations with protection.
"""

import asyncio
from glassbox_rag import GlassBoxEngine
from glassbox_rag.config import load_config


async def main():
    """Run write-back example."""
    # Load configuration
    config = load_config("config/default.yaml")
    
    # Check if write-back is enabled
    if not config.writeback.enabled:
        print("Write-back is disabled in config")
        return
    
    # Create engine
    engine = GlassBoxEngine(config)
    
    # Example 1: Update with high confidence (should succeed)
    print("Example 1: High confidence update")
    print("-" * 40)
    
    result = await engine.update(
        document_id="doc_001",
        content="Updated content with high confidence",
        confidence_score=0.95,  # 95% confidence
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Requires Review: {result.requires_review}")
    print()
    
    # Example 2: Update with low confidence (may require review)
    print("Example 2: Low confidence update")
    print("-" * 40)
    
    result = await engine.update(
        document_id="doc_002",
        content="Updated content with low confidence",
        confidence_score=0.6,  # 60% confidence
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Requires Review: {result.requires_review}")
    if result.review_url:
        print(f"Review URL: {result.review_url}")


if __name__ == "__main__":
    asyncio.run(main())
