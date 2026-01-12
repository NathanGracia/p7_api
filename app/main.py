from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.inference import ModelService
import logging
import os
from azure.monitor.opentelemetry import configure_azure_monitor

# Instancie l'application FastAPI
app = FastAPI(title="Tweet Sentiment API (Scikit-Learn)", version="1.1.0")

# MISE À JOUR : On utilise le modèle Logistique et le Vectoriseur TF-IDF
model_service = ModelService(
    model_path="models/logistic_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
)
# 1. Activation d'Azure Monitor (il lira la clé dans Heroku tout seul)
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if connection_string:
    configure_azure_monitor(connection_string=connection_string)

logger = logging.getLogger("FeedbackLogger")

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        # La logique reste la même : le service renvoie (label, proba)
        pred_label, pred_proba = model_service.predict(payload.text)

        return PredictResponse(
            is_positive=bool(pred_label),  # 1 (Positif) -> True, 0 (Négatif) -> False
            score=float(pred_proba)        # Score de confiance du modèle
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Très utile pour le debug en local si ça plante
        print(f"Erreur d'inférence : {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/feedback")
async def post_feedback(data: dict):
    # Si le notebook envoie is_correct = False
    if data.get("is_correct") is False:
        # On envoie le message NEGATIVE_FEEDBACK à Azure
        logger.warning("NEGATIVE_FEEDBACK", extra={
            "custom_dimensions": {
                "tweet": data.get("text"),
                "prediction": str(data.get("prediction"))
            }
        })
        return {"status": "Alerte logguée sur Azure"}

    return {"status": "Feedback positif reçu"}