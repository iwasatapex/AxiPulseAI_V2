import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)


from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


checks = []


r = client.get("/health")

checks.append(
    r.status_code == 200
)


r = client.get("/openapi.json")

checks.append(
    r.status_code == 200
)


r = client.post(
    "/api/v1/auth/token",
    json={
        "username":"release",
        "password":"test"
    }
)

checks.append(
    r.status_code == 200
)


for c in checks:
    print(
        "PASS" if c else "FAIL"
    )


assert all(checks)

print(
    "SMOKE TEST PASS"
)
