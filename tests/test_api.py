from fastapi.testclient import TestClient
from src.main import app


def test_health_check():
  with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_application():
  payload = {
      "AMT_INCOME_TOTAL": 150000.0,
      "AMT_CREDIT": 500000.0,
      "AMT_ANNUITY": 25000.0,
      "DAYS_BIRTH": -12000,
      "DAYS_EMPLOYED": -1500,
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_2": 0.6,
      "EXT_SOURCE_3": 0.4,
      "NAME_CONTRACT_TYPE": "Cash loans",
      "CODE_GENDER": "F",
      "FLAG_OWN_CAR": "N",
  }
  with TestClient(app) as client:
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["risk_prediction"] in [0, 1]


def test_predict_invalid_income():
  #Checking for negative numbers
  payload = {
      "AMT_INCOME_TOTAL": -5000.0,
      "AMT_CREDIT": 500000.0,
      "DAYS_BIRTH": -12000,
      "DAYS_EMPLOYED": -1500,
      "NAME_CONTRACT_TYPE": "Cash loans",
      "CODE_GENDER": "F",
      "FLAG_OWN_CAR": "N",
  }
  with TestClient(app) as client:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error