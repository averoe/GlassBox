"""
JWT authentication middleware.

Supports JWT token validation and optional OAuth2 integration
for production API security beyond simple API keys.
"""

import time
import hmac
import hashlib
import base64
import json
from typing import Any, Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from glassbox_rag.utils.logging import get_logger

logger = get_logger(__name__)


class AuthError(Exception):
    """Raised on authentication failure."""


class JWTAuth:
    """
    Simple JWT authentication middleware.

    Validates HS256 signed tokens with configurable secret,
    issuer, and expiration checking.
    """

    def __init__(
        self,
        secret: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        leeway_seconds: int = 30,
    ):
        self.secret = secret
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.leeway = leeway_seconds

    def create_token(
        self,
        subject: str,
        expires_in: int = 3600,
        claims: Optional[Dict] = None,
    ) -> str:
        """
        Create a signed JWT token.

        Args:
            subject: Token subject (user ID).
            expires_in: Expiration in seconds from now.
            claims: Additional claims to include.

        Returns:
            Signed JWT string.
        """
        now = int(time.time())
        payload: Dict[str, Any] = {
            "sub": subject,
            "iat": now,
            "exp": now + expires_in,
        }
        if self.issuer:
            payload["iss"] = self.issuer
        if self.audience:
            payload["aud"] = self.audience
        if claims:
            payload.update(claims)

        return self._encode(payload)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode a JWT token.

        Raises:
            AuthError: If token is invalid, expired, or tampered with.
        """
        try:
            payload = self._decode(token)
        except Exception as e:
            raise AuthError(f"Invalid token format: {e}")

        now = int(time.time())

        # Check expiration
        exp = payload.get("exp")
        if exp and now > exp + self.leeway:
            raise AuthError("Token expired")

        # Check not-before
        nbf = payload.get("nbf")
        if nbf and now < nbf - self.leeway:
            raise AuthError("Token not yet valid")

        # Check issuer
        if self.issuer and payload.get("iss") != self.issuer:
            raise AuthError(f"Invalid issuer: {payload.get('iss')}")

        # Check audience
        if self.audience:
            aud = payload.get("aud")
            if isinstance(aud, list):
                if self.audience not in aud:
                    raise AuthError("Invalid audience")
            elif aud != self.audience:
                raise AuthError("Invalid audience")

        return payload

    def _encode(self, payload: Dict) -> str:
        """Encode and sign a JWT (HS256)."""
        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = _b64url_encode(json.dumps(header).encode())
        payload_b64 = _b64url_encode(json.dumps(payload).encode())

        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        signature_b64 = _b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def _decode(self, token: str) -> Dict:
        """Decode and verify a JWT (HS256)."""
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthError("Token must have 3 parts")

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            self.secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise AuthError("Invalid signature")

        payload = json.loads(_b64url_decode(payload_b64))
        return payload


class RedisRateLimiter:
    """
    Distributed rate limiter using Redis sliding window.

    Works across multiple worker processes, unlike
    the in-memory RateLimiter.

    Requires: pip install redis
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_requests: int = 60,
        window_seconds: int = 60,
        prefix: str = "glassbox:ratelimit:",
    ):
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window = window_seconds
        self.prefix = prefix
        self._client = None

    async def initialize(self) -> bool:
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self.redis_url, decode_responses=True,
            )
            await self._client.ping()
            logger.info("Redis rate limiter connected: %s", self.redis_url)
            return True
        except ImportError:
            logger.warning("redis package not installed for rate limiter")
            return False
        except Exception as e:
            logger.warning("Redis rate limiter unavailable: %s", e)
            return False

    async def is_allowed(self, client_key: str) -> bool:
        """Check if client_key is within rate limit."""
        if not self._client:
            return True  # Fail open if Redis unavailable

        key = f"{self.prefix}{client_key}"
        now = time.time()
        window_start = now - self.window

        pipe = self._client.pipeline()
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current entries
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry
        pipe.expire(key, self.window + 1)

        results = await pipe.execute()
        current_count = results[1]

        if current_count >= self.max_requests:
            # Remove the entry we just added
            await self._client.zrem(key, str(now))
            return False

        return True

    async def get_remaining(self, client_key: str) -> int:
        """Get remaining requests in window."""
        if not self._client:
            return self.max_requests

        key = f"{self.prefix}{client_key}"
        now = time.time()
        window_start = now - self.window

        await self._client.zremrangebyscore(key, 0, window_start)
        count = await self._client.zcard(key)
        return max(0, self.max_requests - count)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()


def create_auth_middleware(
    jwt_secret: Optional[str] = None,
    jwt_issuer: Optional[str] = None,
    api_keys: Optional[List[str]] = None,
    excluded_paths: Optional[List[str]] = None,
):
    """
    Create a FastAPI middleware that checks JWT or API key auth.

    Skips auth for excluded_paths (e.g., /health, /docs).
    """
    jwt_auth = JWTAuth(secret=jwt_secret or "change-me", issuer=jwt_issuer) if jwt_secret else None
    excluded = set(excluded_paths or ["/health", "/docs", "/openapi.json", "/redoc"])

    async def auth_middleware(request: Request, call_next):
        if request.url.path in excluded:
            return await call_next(request)

        # Try JWT first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and jwt_auth:
            token = auth_header[7:]
            try:
                claims = jwt_auth.validate_token(token)
                request.state.user = claims
                return await call_next(request)
            except AuthError as e:
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"JWT auth failed: {e}"},
                )

        # Fall back to API key
        api_key = request.headers.get("X-API-Key", "")
        if api_keys and api_key in api_keys:
            request.state.user = {"sub": "api_key", "key": api_key[:8] + "..."}
            return await call_next(request)

        # No valid auth
        if jwt_auth or api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        return await call_next(request)

    return auth_middleware


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)
