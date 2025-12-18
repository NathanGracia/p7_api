from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Vérifie que l'API répond sur /health"""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] == "ok"  # ou "healthy" selon votre code exact

def test_predict_validation():
    """Vérifie que la validation d'entrée fonctionne"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    body = response.json()
    # Vérifie que le message d'erreur est explicite
    assert "error" in body or "detail" in body
