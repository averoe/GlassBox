# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | ✅ Active support  |
| 0.2.x   | ⚠️ Critical fixes only |
| < 0.2   | ❌ No support      |

## Reporting a Vulnerability

If you discover a security vulnerability in GlassBox RAG, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. Email your report to the maintainers:
   - Include a description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment** within 48 hours
- **Assessment** within 5 business days
- **Fix or mitigation** within 30 days for critical issues

### Scope

The following are in scope for security reports:

- Authentication bypass (JWT, API keys)
- API key exposure in logs or error messages
- Rate limiter bypass
- Injection attacks (SQL, code execution)
- Denial of service via resource exhaustion
- Insecure defaults in production configurations

### Known Considerations

- **API keys in config:** Use environment variables (`${OPENAI_API_KEY}`) instead of plaintext in YAML files. We recommend using Pydantic `SecretStr` for sensitive fields.
- **Rate limiting:** The in-memory rate limiter is per-process. In multi-worker deployments, use `rate_limit_backend: redis` for accurate rate limiting.
- **CORS:** The default CORS configuration allows all origins (`*`). Restrict this in production.

## Security Best Practices

When deploying GlassBox RAG in production:

1. Set `security.api_key_required: true` and configure API keys
2. Enable JWT authentication for user-facing endpoints
3. Use `rate_limit_backend: redis` with `workers > 1`
4. Restrict CORS origins to your domain
5. Store all API keys and secrets in environment variables
6. Run behind a reverse proxy (nginx, Caddy) with TLS
7. Set `server.debug: false` in production
