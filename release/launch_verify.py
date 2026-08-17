import sys
import os
from pathlib import Path


# load production environment before importing api
env = Path(__file__).resolve().parent.parent / ".env"

if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line:
            k,v = line.split("=",1)
            os.environ[k] = v


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)


from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


checks = {}


# API

r = client.get("/health")

checks["health"] = (
    r.status_code == 200
)



# OpenAPI

r = client.get("/openapi.json")

checks["openapi"] = (
    r.status_code == 200
)



# Authentication

r = client.post(
    "/api/v1/auth/token",
    json={
        "username":"production",
        "password":"verify"
    }
)

checks["authentication"] = (
    r.status_code == 200
    and
    "access_token" in r.json()
)



# Metrics

r = client.get("/metrics")

checks["monitoring"] = (
    r.status_code == 200
)



# ADIE

token = client.post(
    "/api/v1/auth/token",
    json={
        "username":"production",
        "password":"verify"
    }
).json()["access_token"]


r = client.post(
    "/api/v1/adie/decision",
    headers={
        "Authorization":
        f"Bearer {token}",
        "X-API-Key":
        __import__("os").getenv("AXIPULSE_API_KEY")
    },
    json={
        "state":{
            "timeline":[
                {
                    "operations_health":82,
                    "competency":88,
                    "quality":85,
                    "attendance":90,
                    "release":60,
                    "transfer":14,
                    "nps":88
                }
            ]
        },
        "recommendations":[]
    }
)

checks["adie"] = (
    r.status_code == 200
)



for k,v in checks.items():

    print(
        k,
        "PASS" if v else "FAIL"
    )


assert all(checks.values())


print()
print(
    "AxiPulseAI v1.0 PRODUCTION READY"
)
