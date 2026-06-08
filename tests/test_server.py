"""Tests for evaluation metrics and the FastAPI server (using the mock backend).

The mock backend lets us exercise the full HTTP stack - request validation,
streaming, error handling - without loading any model or needing a GPU.
"""

from __future__ import annotations

import os

import pytest

from src.evaluate import keyword_em, normalize, token_f1

# ---------- Metric tests ----------


def test_normalize_strips_punctuation_and_articles():
    assert normalize("The Carmack Amendment, 1906.") == "carmack amendment 1906"
    assert normalize("  multiple   spaces   ") == "multiple spaces"


def test_token_f1_perfect_match():
    assert token_f1("the carmack amendment", "Carmack Amendment") == pytest.approx(1.0)


def test_token_f1_no_overlap_is_zero():
    assert token_f1("apples oranges", "x y z") == 0.0


def test_token_f1_partial():
    f1 = token_f1("freight class is determined by NMFC density", "NMFC density and stowability")
    # Some overlap ("nmfc", "density") → should be > 0 but < 1
    assert 0.0 < f1 < 1.0


def test_keyword_em_all_present():
    pred = "Concealed damage claims must be filed within 9 months under 49 CFR 370."
    facts = ["concealed damage", "9 months", "49 CFR"]
    assert keyword_em(pred, facts) == pytest.approx(1.0)


def test_keyword_em_partial():
    pred = "Claims must be filed within 9 months."
    facts = ["concealed damage", "9 months", "49 CFR"]
    assert keyword_em(pred, facts) == pytest.approx(1 / 3)


def test_keyword_em_empty_facts_is_zero():
    assert keyword_em("anything", []) == 0.0


# ---------- Server tests (mock backend) ----------


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with USE_MOCK enabled so no model is loaded."""
    os.environ["USE_MOCK"] = "true"
    # Import inside the fixture so the env var is picked up by the lifespan.
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"


def test_generate_endpoint_mock(client):
    r = client.post("/generate", json={"question": "What is dimensional weight?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert body["has_adapter"] is False
    assert "MOCK" in body["answer"]


def test_generate_rejects_empty_question(client):
    r = client.post("/generate", json={"question": ""})
    assert r.status_code == 422  # pydantic validation


def test_generate_rejects_excessive_max_tokens(client):
    r = client.post("/generate", json={"question": "Q?", "max_new_tokens": 10000})
    assert r.status_code == 422


def test_stream_endpoint(client):
    """Streaming response should yield multiple chunks and a [DONE] sentinel."""
    with client.stream("POST", "/generate/stream", json={"question": "Test"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
        assert "data:" in body
        assert "[DONE]" in body
