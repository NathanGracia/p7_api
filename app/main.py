from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.inference import ModelService

#  instancie l'application FastAPI
app = FastAPI(title="Tweet Sentiment API", version="1.0.0")

#  crée un service d'inférence (chargement du modèle une seule fois)
model_service = ModelService(model_path="models/model_test.pkl")

@app.get("/health", response_model=HealthResponse)
def health():
    #  fournit un endpoint de santé simple pour les probes Azure
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    #  valide l'entrée et gère les erreurs d'inférence
    try:
        pred_label, pred_proba = model_service.predict(payload.text)
        #  renvoie une sortie typée et stable
        return PredictResponse(
            is_positive=bool(pred_label),
            score=float(pred_proba)
        )
    except ValueError as ve:
        #  renvoie une 400 si l'entrée est invalide (ex: texte vide)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        #  évite d'exposer des détails sensibles en production
        raise HTTPException(status_code=500, detail="Inference failed")
