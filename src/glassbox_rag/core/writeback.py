"""
Write-Back Module - handles safe bi-directional write operations.

Implements configurable protection layers including confidence thresholds,
human review, and read-only mode.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from glassbox_rag.config import GlassBoxConfig


class WriteBackMode(Enum):
    """Write-back operation modes."""

    READ_ONLY = "read-only"
    PROTECTED = "protected"
    FULL = "full"


@dataclass
class WriteBackRequest:
    """Represents a write-back operation request."""

    operation: str  # "create", "update", or "delete"
    document_id: str
    content: str
    metadata: Dict[str, Any]
    confidence_score: float = 1.0
    user_id: Optional[str] = None


@dataclass
class WriteBackResult:
    """Result of a write-back operation."""

    success: bool
    operation_id: str
    message: str
    requires_review: bool = False
    review_url: Optional[str] = None


class WriteBackManager:
    """
    Manages safe bi-directional write operations.

    Enforces protection layers based on configuration:
    - read-only: No writes allowed
    - protected: Requires confidence threshold or human review
    - full: Direct writes allowed
    """

    def __init__(self, config: GlassBoxConfig):
        """Initialize write-back manager."""
        self.config = config
        self.mode = WriteBackMode(config.writeback.mode)
        self.enabled = config.writeback.enabled
        self.protected_config = config.writeback.protected or {}

    async def execute_write(
        self,
        request: WriteBackRequest,
    ) -> WriteBackResult:
        """
        Execute a write-back operation with protection checks.

        Args:
            request: Write-back request.

        Returns:
            WriteBackResult with operation status.
        """
        # Check if write-back is enabled
        if not self.enabled:
            return WriteBackResult(
                success=False,
                operation_id="",
                message="Write-back is disabled",
            )

        # Check mode
        if self.mode == WriteBackMode.READ_ONLY:
            return WriteBackResult(
                success=False,
                operation_id="",
                message="Database is in read-only mode",
            )

        # Apply protection rules if in PROTECTED mode
        if self.mode == WriteBackMode.PROTECTED:
            return await self._execute_protected_write(request)

        # Full mode - direct write
        return await self._execute_write_directly(request)

    async def _execute_protected_write(
        self,
        request: WriteBackRequest,
    ) -> WriteBackResult:
        """Execute write with protection checks."""
        confidence_threshold = self.protected_config.get("confidence_threshold", 0.8)
        requires_human_review = self.protected_config.get("human_review", False)

        # Check confidence threshold
        if request.confidence_score < confidence_threshold:
            if requires_human_review:
                return await self._create_review_request(request)
            else:
                return WriteBackResult(
                    success=False,
                    operation_id="",
                    message=(
                        f"Confidence score {request.confidence_score} below threshold "
                        f"{confidence_threshold}"
                    ),
                )

        # If human review is required regardless of confidence
        if requires_human_review:
            return await self._create_review_request(request)

        # Confidence is sufficient, proceed with write
        return await self._execute_write_directly(request)

    async def _execute_write_directly(
        self,
        request: WriteBackRequest,
    ) -> WriteBackResult:
        """Execute write operation directly."""
        # TODO: Implement actual database write
        operation_id = f"wb_{request.operation}_{request.document_id}"

        return WriteBackResult(
            success=True,
            operation_id=operation_id,
            message=f"Document {request.document_id} {request.operation}d successfully",
        )

    async def _create_review_request(
        self,
        request: WriteBackRequest,
    ) -> WriteBackResult:
        """Create a human review request for the write operation."""
        review_url = self.protected_config.get("review_url")

        if not review_url:
            return WriteBackResult(
                success=False,
                operation_id="",
                message="Human review required but no review URL configured",
            )

        # TODO: Create review request in database
        operation_id = f"review_{request.operation}_{request.document_id}"

        return WriteBackResult(
            success=False,
            operation_id=operation_id,
            message="Write operation pending human review",
            requires_review=True,
            review_url=review_url,
        )

    def get_mode(self) -> WriteBackMode:
        """Get current write-back mode."""
        return self.mode

    def set_mode(self, mode: WriteBackMode):
        """Set write-back mode."""
        self.mode = mode

    def is_enabled(self) -> bool:
        """Check if write-back is enabled."""
        return self.enabled
