from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)



def test_token_creation():

    response = client.post(
        "/api/v1/auth/token",
        json={
            "username":"test",
            "password":"test"
        }
    )

    assert response.status_code == 200

    assert (
        "access_token"
        in response.json()
    )



def test_invalid_token():

    response = client.get(
        "/api/v1/history/decisions",
        headers={
            "Authorization":
            "Bearer invalid"
        }
    )

    assert response.status_code == 401
