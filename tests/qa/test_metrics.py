from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)



def test_metrics_endpoint():

    client.get("/health")

    r = client.get(
        "/metrics"
    )

    assert r.status_code == 200

    assert (
        "axipulse_api_requests_total"
        in r.text
    )
