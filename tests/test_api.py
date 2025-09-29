from fastapi.testclient import TestClient
from app.main import app

# crée un client de test
client = TestClient(app)

def test_health():
    # vérifie que l'API répond
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_validation():
    # vérifie les erreurs d'entrée
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 400
