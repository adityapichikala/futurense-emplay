"""API tests: health/fields are cheap; /extract uses a single small PDF."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rfp_extractor.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["fields"] == 20


def test_fields_contract(client):
    response = client.get("/fields")
    assert response.status_code == 200
    body = response.json()
    assert body["bid_number"] == "Bid Number"
    assert "product_specification" in body


def test_extract_rejects_empty(client):
    response = client.post("/extract")
    assert response.status_code in (400, 422)


def test_extract_accepts_upload(bid2_dir, client):
    path = bid2_dir / "Mercury_Affidavit.pdf"
    with open(path, "rb") as fh:
        response = client.post("/extract", files={"files": (path.name, fh, "application/pdf")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "packages" in body


def test_extract_rejects_unsupported_type(client, tmp_path):
    bad = tmp_path / "notes.exe"
    bad.write_bytes(b"MZ")
    with open(bad, "rb") as fh:
        response = client.post("/extract", files={"files": ("notes.exe", fh, "application/octet-stream")})
    assert response.status_code == 415
