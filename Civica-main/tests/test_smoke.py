"""Smoke tests for CivicaAI's FastAPI backend and Flask frontend routes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def fastapi_client():
    try:
        from fastapi.testclient import TestClient
        from api import app
    except ImportError as exc:
        pytest.skip(f"FastAPI stack unavailable: {exc}")

    import api

    api.model_info = lambda task: {"provider": "test", "model": "smoke"}
    api.run_scheme = lambda **kwargs: {
        "summary": "smoke",
        "eligibility": [],
        "benefits": [],
        "process": [],
        "sources": kwargs.get("source_urls", []),
    }
    api.run_fraud = lambda **kwargs: {
        "result_class": "Unverified___Needs_Caution",
        "result_text": "smoke",
        "result_details": "smoke",
    }
    api.chat_reply = lambda message, history=None: "smoke"
    return TestClient(app)


def test_fastapi_health_and_openapi(fastapi_client):
    response = fastapi_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    openapi = fastapi_client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/schemes/analyze" in paths
    assert "/api/v1/schemes/analyze-file" in paths
    assert "/api/v1/fraud/analyze" in paths
    assert "/api/v1/chat" in paths


def test_fastapi_validation_and_mocked_business_paths(fastapi_client):
    assert fastapi_client.post("/api/v1/schemes/analyze", json={}).status_code == 400
    assert fastapi_client.post(
        "/api/v1/schemes/analyze", json={"text": "PM scheme"}
    ).status_code == 200
    assert fastapi_client.post("/api/v1/fraud/analyze").status_code == 400
    assert fastapi_client.post("/api/v1/chat", json={"message": "hello"}).status_code == 200


def test_flask_frontend_routes():
    try:
        from app import app
    except ImportError as exc:
        pytest.skip(f"Flask/Google stack unavailable: {exc}")

    client = app.test_client()
    expected = ["/", "/index/", "/home/", "/scheme/", "/fraud/", "/mitra/"]
    for route in expected:
        response = client.get(route)
        assert response.status_code == 200, route
        assert response.mimetype == "text/html", route


def test_frontend_scheme_contract_matches_flask():
    html = (ROOT / "templates" / "scheme.html").read_text(encoding="utf-8")
    assert "fetch('/analyze'" in html
    assert "data-type=\"file\"" in html
    assert "data-type=\"url\"" in html
    assert "data-type=\"text\"" in html


def test_frontend_templates_exist():
    for name in ("index.html", "home.html", "scheme.html", "fraud.html", "mitra.html"):
        assert (ROOT / "templates" / name).is_file(), name
