from app.runtime import resolve_runtime_mode


def test_resolve_runtime_mode_prefers_embedded_when_forced():
    assert resolve_runtime_mode("embedded", api_is_available=True) == "embedded"


def test_resolve_runtime_mode_prefers_api_when_forced():
    assert resolve_runtime_mode("api", api_is_available=False) == "api"


def test_resolve_runtime_mode_falls_back_when_api_unavailable():
    assert resolve_runtime_mode("auto", api_is_available=False) == "embedded"
    assert resolve_runtime_mode("auto", api_is_available=True) == "api"