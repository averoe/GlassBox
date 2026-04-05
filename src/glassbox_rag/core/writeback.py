"""
Write-Back Module — handles safe bi-directional write operations.

Implements configurable protection layers including confidence thresholds,
human review, and read-only mode. Includes input validation and
structured logging.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from uuid import uuid4

from glassbox_rag.config import GlassBoxConfig
from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class WriteBackError(Exception):
    """Raised when a write-back operation fails."""


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    user_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate request fields."""
        valid_ops = {"create", "update", "delete"}
        if self.operation not in valid_ops:
            raise ValueError(
                f"Invalid operation '{self.operation}'. Must be one of {valid_ops}"
            )
        
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )

        if not self.document_id or not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


@dataclass
class WriteBackResult:
    """Result of a write-back operation."""

    success: bool
    operation_id: str
    message: str
    requires_review: bool = False
    review_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation_id": self.operation_id,
            "message": self.message,
            "requires_review": self.requires_review,
            "review_url": self.review_url,
        }


class WriteBackManager:
    """
    Manages safe bi-directional write operations.

    Enforces protection layers based on configuration:
    - read-only: No writes allowed
    - protected: Requires confidence threshold or human review
    - full: Direct writes allowed
    """

    def __init__(self, config: GlassBoxConfig, database: Any = None):
        self.config = config
        self.mode = WriteBackMode(config.writeback.mode)
        self.enabled = config.writeback.enabled
        self.protected_config = config.writeback.protected or {}
        self.database = database

        # Pending review queue (in-memory for now)
        self._pending_reviews: Dict[str, WriteBackRequest] = {}

        logger.info("WriteBack manager initialized: mode=%s, enabled=%s", self.mode.value, self.enabled)

    async def execute_write(self, request: WriteBackRequest) -> WriteBackResult:
        """
        Execute a write-back operation with protection checks.

        Args:
            request: Write-back request.

        Returns:
            WriteBackResult with operation status.

        Raises:
            WriteBackError: If the write operation encounters an error.
        """
        operation_id = str(uuid4())

        # Check if write-back is enabled
        if not self.enabled:
            logger.info("Write-back disabled, rejecting %s on %s", request.operation, request.document_id)
            return WriteBackResult(
                success=False,
                operation_id=operation_id,
                message="Write-back is disabled",
            )

        # Check mode
        if self.mode == WriteBackMode.READ_ONLY:
            logger.info("Read-only mode, rejecting %s on %s", request.operation, request.document_id)
            return WriteBackResult(
                success=False,
                operation_id=operation_id,
                message="Database is in read-only mode",
            )

        # Apply protection rules if in PROTECTED mode
        if self.mode == WriteBackMode.PROTECTED:
            return await self._execute_protected_write(request, operation_id)

        # Full mode — direct write
        return await self._execute_write_directly(request, operation_id)

    async def _execute_protected_write(
        self,
        request: WriteBackRequest,
        operation_id: str,
    ) -> WriteBackResult:
        """Execute write with protection checks."""
        confidence_threshold = self.protected_config.get("confidence_threshold", 0.8)
        requires_human_review = self.protected_config.get("human_review", False)

        logger.debug(
            "Protected write: op=%s, doc=%s, confidence=%.2f (threshold=%.2f, review=%s)",
            request.operation,
            request.document_id,
            request.confidence_score,
            confidence_threshold,
            requires_human_review,
        )

        # Check confidence threshold
        if request.confidence_score < confidence_threshold:
            if requires_human_review:
                return await self._create_review_request(request, operation_id)
            else:
                logger.warning(
                    "Write rejected: confidence %.2f below threshold %.2f for doc %s",
                    request.confidence_score,
                    confidence_threshold,
                    request.document_id,
                )
                return WriteBackResult(
                    success=False,
                    operation_id=operation_id,
                    message=(
                        f"Confidence score {request.confidence_score:.2f} below threshold "
                        f"{confidence_threshold:.2f}"
                    ),
                )

        # If human review is required regardless of confidence
        if requires_human_review:
            return await self._create_review_request(request, operation_id)

        # Confidence is sufficient, proceed with write
        return await self._execute_write_directly(request, operation_id)

    async def _execute_write_directly(
        self,
        request: WriteBackRequest,
        operation_id: str,
    ) -> WriteBackResult:
        """Execute write operation directly against the database."""
        try:
            if self.database is not None:
                if request.operation == "create":
                    await self.database.insert(
                        "documents",
                        {
                            "id": request.document_id,
                            "content": request.content,
                            "metadata": request.metadata,
                        },
                    )
                elif request.operation == "update":
                    await self.database.update(
                        "documents",
                        request.document_id,
                        {"content": request.content, "metadata": request.metadata},
                    )
                elif request.operation == "delete":
                    await self.database.delete("documents", request.document_id)

            logger.info(
                "Write-back success: op=%s, doc=%s, operation_id=%s",
                request.operation,
                request.document_id,
                operation_id,
            )

            return WriteBackResult(
                success=True,
                operation_id=operation_id,
                message=f"Document {request.document_id} {request.operation}d successfully",
            )

        except Exception as e:
            logger.error(
                "Write-back failed: op=%s, doc=%s, error=%s",
                request.operation,
                request.document_id,
                e,
            )
            raise WriteBackError(
                f"Failed to {request.operation} document {request.document_id}: {e}"
            ) from e

    async def _create_review_request(
        self,
        request: WriteBackRequest,
        operation_id: str,
    ) -> WriteBackResult:
        """Create a human review request for the write operation."""
        review_url = self.protected_config.get("review_url")

        # Store in pending reviews
        self._pending_reviews[operation_id] = request

        logger.info(
            "Write-back queued for review: op=%s, doc=%s, operation_id=%s",
            request.operation,
            request.document_id,
            operation_id,
        )

        return WriteBackResult(
            success=False,
            operation_id=operation_id,
            message="Write operation pending human review",
            requires_review=True,
            review_url=review_url,
        )

    async def approve_review(self, operation_id: str) -> WriteBackResult:
        """Approve a pending review and execute the write."""
        request = self._pending_reviews.pop(operation_id, None)
        if request is None:
            return WriteBackResult(
                success=False,
                operation_id=operation_id,
                message=f"No pending review found for operation {operation_id}",
            )

        logger.info("Review approved: operation_id=%s", operation_id)
        return await self._execute_write_directly(request, operation_id)

    async def reject_review(self, operation_id: str) -> WriteBackResult:
        """Reject a pending review."""
        request = self._pending_reviews.pop(operation_id, None)
        if request is None:
            return WriteBackResult(
                success=False,
                operation_id=operation_id,
                message=f"No pending review found for operation {operation_id}",
            )

        logger.info("Review rejected: operation_id=%s", operation_id)
        return WriteBackResult(
            success=False,
            operation_id=operation_id,
            message="Write operation rejected by reviewer",
        )

    def get_pending_reviews(self) -> Dict[str, Dict]:
        """Get all pending review requests."""
        return {
            op_id: {
                "operation": req.operation,
                "document_id": req.document_id,
                "confidence_score": req.confidence_score,
                "user_id": req.user_id,
            }
            for op_id, req in self._pending_reviews.items()
        }

    def get_mode(self) -> WriteBackMode:
        """Get current write-back mode."""
        return self.mode

    def set_mode(self, mode: WriteBackMode) -> None:
        """Set write-back mode."""
        old_mode = self.mode
        self.mode = mode
        logger.info("WriteBack mode changed: %s → %s", old_mode.value, mode.value)

    def is_enabled(self) -> bool:
        """Check if write-back is enabled."""
        return self.enabled
