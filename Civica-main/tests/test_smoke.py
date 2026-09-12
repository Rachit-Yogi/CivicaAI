"""Smoke tests for CivicaAI's FastAPI application and web UI."""
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


def test_web_ui_routes(fastapi_client):
    for route in ("/", "/index/", "/home/", "/scheme/", "/fraud/", "/mitra/"):
        response = fastapi_client.get(route)
        assert response.status_code == 200, route
        assert response.headers["content-type"].startswith("text/html"), route


def test_static_files_are_served(fastapi_client):
    response = fastapi_client.get("/static/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_fastapi_validation_and_mocked_business_paths(fastapi_client):
    assert fastapi_client.post("/api/v1/schemes/analyze", json={}).status_code == 400
    assert fastapi_client.post(
        "/api/v1/schemes/analyze", json={"text": "PM scheme"}
    ).status_code == 200
    assert fastapi_client.post("/api/v1/fraud/analyze").status_code == 400
    assert fastapi_client.post("/api/v1/chat", json={"message": "hello"}).status_code == 200


def test_frontend_contracts():
    scheme = (ROOT / "templates" / "scheme.html").read_text(encoding="utf-8")
    mitra = (ROOT / "templates" / "mitra.html").read_text(encoding="utf-8")
    fraud = (ROOT / "templates" / "fraud.html").read_text(encoding="utf-8")

    assert "/api/v1/schemes/analyze-file" in scheme
    assert "/api/v1/schemes/analyze" in scheme
    assert "/api/v1/chat" in mitra
    assert "/api/v1/fraud/analyze" in fraud
    assert "url_for(" not in fraud
    assert "/analyze" not in scheme.replace("/api/v1/schemes/analyze", "")


def test_no_flask_dependency():
    requirements = (ROOT.parent / "requirements.txt").read_text(encoding="utf-8")
    assert "flask" not in requirements.lower()
    assert "fastapi" in requirements.lower()


def test_frontend_templates_exist():
    for name in ("index.html", "home.html", "scheme.html", "fraud.html", "mitra.html"):
        assert (ROOT / "templates" / name).is_file(), name
