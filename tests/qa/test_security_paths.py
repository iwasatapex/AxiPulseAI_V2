from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)



def test_missing_auth():

    r = client.get(
        "/api/v1/history/decisions"
    )

    assert r.status_code in [
        401,
        403
    ]



def test_bad_token():

    r = client.get(
        "/api/v1/history/predictions",
        headers={
            "Authorization":
            "Bearer bad-token"
        }
    )

    assert r.status_code == 401
