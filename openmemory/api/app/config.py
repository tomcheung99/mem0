import os


def _normalize_openai_environment() -> None:
    gateway_api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    gateway_base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()

    if gateway_api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = gateway_api_key

    if gateway_base_url and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = gateway_base_url


def _parse_allowed_origins(value: str | None) -> list[str]:
    if not value:
        return ["*"]

    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["*"]


_normalize_openai_environment()


USER_ID = os.getenv("USER", "default_user")
DEFAULT_APP_ID = "openmemory"
MCP_API_KEY = os.getenv("MCP_API_KEY", "").strip()
MCP_RATE_LIMIT = os.getenv("MCP_RATE_LIMIT", "100/minute").strip() or "100/minute"
ALLOWED_ORIGINS = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
ALLOW_CREDENTIALS = ALLOWED_ORIGINS != ["*"]