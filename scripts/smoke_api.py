# scripts/smoke_api.py
import json
import os
import urllib.request

API = os.getenv("API_BASE", "http://localhost:8080")


def get(path):
    with urllib.request.urlopen(f"{API}{path}") as r:
        return r.read().decode()


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


print("API_BASE:", API)

# 1) healthz (non-blocking)
print("GET /healthz ->", get("/healthz"))

# 2) readyz (bounded load)
print("GET /readyz ->", get("/readyz"))

# 3) predict (minimal valid payload)
sample = {
    "rows": [
        {
            "type": "PAYMENT",
            "amount": 123.45,
            "step": 1,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 876.55,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 623.45,
        }
    ]
}
print("POST /predict ->", post("/predict", sample))

print("SMOKE API: OK")
