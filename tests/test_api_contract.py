#tests/test_api_contract.py
from fastapi.testclient import TestClient

from mlops_fraud.deployment.api import app

client = TestClient(app)

def test_rejects_unknown_field():
    bad = {
        "rows": [{
            "type": "PAYMENT",
            "amount": 10,
            "step": 1,
            "oldbalanceOrg": 100,
            "newbalanceOrig": 90,
            "oldbalanceDest": 50,
            "newbalanceDest": 60,
            "UNKNOWN": "boom"          # should be rejected
        }]
    }
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "Invalid request"
    # Pydantic reports path to offending field
    assert any("UNKNOWN" in str(item["loc"]) for item in body["detail"])
