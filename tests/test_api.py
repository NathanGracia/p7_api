from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    """Test du endpoint /health - doit renvoyer 200 et status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] == "ok"


def test_predict_validation():
    """Test de validation - un texte vide doit renvoyer une erreur 400."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    body = response.json()
    assert "error" in body or "detail" in body
