# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in AST-RAG, please report it privately.

### How to Report

**Option 1: GitHub Security Advisory (Recommended)**

1. Go to [Security tab](https://github.com/lexasub/raged/security)
2. Click "Report a vulnerability"
3. Provide details (this is private and not visible to public)

**Option 2: Email**

If you cannot use GitHub, email: **your-email@example.com**

### What to Include

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** (what can an attacker do?)
- **Suggested fix** (if you have one)
- **Your GitHub username** (for credit, optional)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Status update:** Within 5 business days
- **Resolution:** Depends on severity (critical issues prioritized)

### Security Best Practices for Users

AST-RAG is designed for **internal use** only. For production deployments:

1. **Change default passwords** in Neo4j and Qdrant
2. **Use HTTPS** for all database connections
3. **Restrict network access** to databases (firewall, VPC)
4. **Keep dependencies updated** (`pip install --upgrade ast-rag`)
5. **Review embedding server** access controls

### Example Secure Configuration

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "STRONG_PASSWORD"
  },
  "qdrant": {
    "url": "http://localhost:6333",
    "api_key": "YOUR_API_KEY"
  }
}
```

### Known Limitations

- Not recommended for public-facing services without additional security layers
- Embedding server should be on private network
- No built-in authentication for MCP server

## Security Updates

Security patches are released as patch versions (e.g., 0.3.1, 0.3.2).

Subscribe to [Releases](https://github.com/lexasub/raged/releases) for notifications.
