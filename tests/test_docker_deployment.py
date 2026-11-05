from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from specrag.interfaces.mcp import validate_mcp_bind_host


ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_compose_preserves_process_ownership_and_private_mcp() -> None:
    compose = _compose()
    services = compose["services"]
    assert set(services) == {"api", "mcp", "web", "cli"}

    api = services["api"]
    mcp = services["mcp"]
    web = services["web"]
    assert api["env_file"] == [".env"]
    assert mcp["env_file"] == [".env"]
    assert api["command"][-2:] == ["--worker-mode", "conversation"]
    assert api["environment"]["SPECRAG_MCP_URL"] == "http://mcp:8766/mcp"
    assert api["depends_on"]["mcp"]["condition"] == "service_healthy"
    assert api["ports"] == ["127.0.0.1:8000:8000"]

    assert "--allow-non-loopback" in mcp["command"]
    assert mcp["expose"] == ["8766"]
    assert "ports" not in mcp
    assert web["ports"] == ["127.0.0.1:5173:80"]
    assert web["depends_on"]["api"]["condition"] == "service_healthy"
    assert all("healthcheck" in services[name] for name in ("api", "mcp", "web"))


def test_compose_uses_named_state_and_read_only_runtime_corpus() -> None:
    services = _compose()["services"]
    for name in ("api", "mcp"):
        volumes = services[name]["volumes"]
        assert "specrag_local:/app/.local" in volumes
        assert "specrag_model_cache:/home/specrag/.cache" in volumes
        assert "./data/3gpp:/app/data/3gpp:ro" in volumes

    cli = services["cli"]
    assert cli["profiles"] == ["tools"]
    assert cli["entrypoint"] == ["specrag"]
    assert "./data/3gpp:/app/data/3gpp" in cli["volumes"]
    assert "./data/3gpp:/app/data/3gpp:ro" not in cli["volumes"]


def test_container_build_excludes_secrets_and_runtime_data() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", ".local", "artifacts", "data/3gpp"):
        assert required in dockerignore

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS backend" in dockerfile
    assert "AS web" in dockerfile
    assert "USER specrag" in dockerfile
    assert "COPY .env" not in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "SPECRAG_MCP_DELEGATION_SECRET=" in example


def test_web_proxy_supports_api_health_and_sse() -> None:
    nginx = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "location /api/" in nginx
    assert "location = /healthz" in nginx
    assert "proxy_pass http://api:8000" in nginx
    assert "proxy_buffering off" in nginx
    assert "proxy_read_timeout 3600s" in nginx
    assert "try_files $uri $uri/ /index.html" in nginx


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "::1"))
def test_mcp_loopback_bind_remains_allowed(host: str) -> None:
    validate_mcp_bind_host(host)


def test_mcp_non_loopback_bind_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="--allow-non-loopback"):
        validate_mcp_bind_host("0.0.0.0")

    validate_mcp_bind_host("0.0.0.0", allow_non_loopback=True)
