"""Testes da API (executar na raiz do projecto: `pytest`)."""
import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)
P = "/api/v1"
FULL = {"age": 55, "sex_male": 0, "height_cm": 160, "weight_kg": 85, "waist_cm": 102,
        "sbp": 150, "dbp": 95, "told_hypertension": 1}


def test_health_and_models():
    assert client.get(f"{P}/health").json()["status"] == "ok"
    assert client.get(f"{P}/models").json()["total_models"] == 4


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "Rastreio de risco de diabetes" in r.text


@pytest.mark.parametrize("model", ["lr", "rf", "lgbm", "xgb"])
def test_predict_ranks_risky_profile_higher(model):
    risky = client.post(f"{P}/predict", json={"model_name": model, "features": FULL}).json()["result"]
    low = client.post(f"{P}/predict", json={"model_name": model, "features": {"age": 28, "sex_male": 1}}).json()["result"]
    assert 0 <= low["risk_score"] < risky["risk_score"] <= 1
    assert risky["risk_level"] in {"medium", "high"}


def test_age_outside_training_range_rejected():
    r = client.post(f"{P}/predict", json={"model_name": "lr", "features": {"age": 20, "sex_male": 1}})
    assert r.status_code == 422


def test_explanation_only_when_requested():
    base = {"model_name": "lr", "features": FULL}
    assert client.post(f"{P}/predict", json=base).json()["explanation"] is None
    ex = client.post(f"{P}/predict", json={**base, "explain": True}).json()["explanation"]
    keys = {f["key"] for f in ex["factors"]}
    assert {"age", "body", "bp", "tobacco"} <= keys
    top = ex["factors"][0]
    assert top["key"] == "body" and top["effect"] > 0          # cintura/peso elevados aumentam o risco
    assert not next(f for f in ex["factors"] if f["key"] == "diet")["provided"]
    assert next(f for f in ex["factors"] if f["key"] == "diet")["effect"] == 0


def test_batch():
    r = client.post(f"{P}/predict/batch", json={"model_name": "lr", "items": [{"features": FULL}, {"features": {"age": 30, "sex_male": 1}}]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2
