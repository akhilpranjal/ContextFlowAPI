from __future__ import annotations


def resolve_runtime_mode(runtime_mode: str, api_is_available: bool | None) -> str:
    """Resolve the active runtime from configuration and API reachability."""

    normalized_mode = runtime_mode.strip().lower()
    if normalized_mode == "embedded":
        return "embedded"
    if normalized_mode == "api":
        return "api"
    return "api" if api_is_available else "embedded"