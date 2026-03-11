import os


def _parse_allowed_origins(value: str | None) -> list[str]:
    if not value:
        return ["*"]

    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["*"]


USER_ID = os.getenv("USER", "default_user")
DEFAULT_APP_ID = "openmemory"
MCP_API_KEY = os.getenv("MCP_API_KEY", "").strip()
MCP_RATE_LIMIT = os.getenv("MCP_RATE_LIMIT", "100/minute").strip() or "100/minute"
ALLOWED_ORIGINS = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
ALLOW_CREDENTIALS = ALLOWED_ORIGINS != ["*"]