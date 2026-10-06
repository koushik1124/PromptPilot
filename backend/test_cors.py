"""
Unit and Integration Tests for Production CORS Hardening (E2-1)
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.main import (
    DEFAULT_DEV_ORIGINS,
    create_app,
    get_allowed_origins,
    parse_allowed_origins,
)


def test_parse_allowed_origins_empty_and_none():
    assert parse_allowed_origins(None) == []
    assert parse_allowed_origins("") == []
    assert parse_allowed_origins("   ") == []


def test_parse_allowed_origins_comma_separated_and_whitespace():
    raw = "  https://example.com , chrome-extension://abc123xyz , , https://app.promptpilot.dev  "
    parsed = parse_allowed_origins(raw)
    assert parsed == [
        "https://example.com",
        "chrome-extension://abc123xyz",
        "https://app.promptpilot.dev",
    ]


def test_dev_mode_defaults_to_local_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    origins = get_allowed_origins()
    assert origins == DEFAULT_DEV_ORIGINS
    assert "http://localhost" in origins
    assert "http://127.0.0.1" in origins


def test_dev_mode_custom_allowed_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:4000,http://127.0.0.1:4000")
    origins = get_allowed_origins()
    assert origins == ["http://localhost:4000", "http://127.0.0.1:4000"]


def test_prod_mode_requires_allowed_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS environment variable must be set in production mode"):
        get_allowed_origins()


def test_prod_mode_rejects_empty_allowed_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "  ,  ")
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS environment variable must be set in production mode"):
        get_allowed_origins()


def test_prod_mode_rejects_wildcard(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="Wildcard '\\*' CORS origin is not permitted in production mode"):
        get_allowed_origins()

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com, *")
    with pytest.raises(RuntimeError, match="Wildcard '\\*' CORS origin is not permitted in production mode"):
        get_allowed_origins()


def test_prod_mode_accepts_explicit_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://app.promptpilot.com,chrome-extension://hgnkghfklaabcdef",
    )
    origins = get_allowed_origins()
    assert origins == [
        "https://app.promptpilot.com",
        "chrome-extension://hgnkghfklaabcdef",
    ]


def test_cors_middleware_allowed_and_disallowed_origins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://app.promptpilot.com,chrome-extension://testextensionid",
    )

    test_app = create_app()
    client = TestClient(test_app)

    # 1. Allowed web origin
    res = client.get(
        "/health",
        headers={"Origin": "https://app.promptpilot.com"},
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://app.promptpilot.com"

    # 2. Allowed Chrome extension origin
    res_ext = client.get(
        "/health",
        headers={"Origin": "chrome-extension://testextensionid"},
    )
    assert res_ext.status_code == 200
    assert res_ext.headers.get("access-control-allow-origin") == "chrome-extension://testextensionid"

    # 3. Disallowed untrusted origin
    res_untrusted = client.get(
        "/health",
        headers={"Origin": "https://malicious-site.com"},
    )
    assert res_untrusted.status_code == 200
    assert "access-control-allow-origin" not in res_untrusted.headers


def test_cors_preflight_options_request(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://app.promptpilot.com",
    )

    test_app = create_app()
    client = TestClient(test_app)

    # Preflight for allowed origin
    res = client.options(
        "/api/analyze",
        headers={
            "Origin": "https://app.promptpilot.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://app.promptpilot.com"

    # Preflight for disallowed origin
    res_bad = client.options(
        "/api/analyze",
        headers={
            "Origin": "https://unauthorized.org",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res_bad.headers.get("access-control-allow-origin") != "https://unauthorized.org"
