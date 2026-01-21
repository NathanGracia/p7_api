from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.inference import ModelService
import logging
import os
from azure.monitor.opentelemetry import configure_azure_monitor

# Instance principale de l'appli FastAPI
app = FastAPI(title="Tweet Sentiment API (Scikit-Learn)", version="1.1.0")

# Chargement du modèle logistique et du vectoriseur TF-IDF
model_service = ModelService(
    model_path="models/logistic_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
)

# Config Azure Monitor - récupère la clé depuis les variables d'environnement
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
        pred_label, pred_proba = model_service.predict(payload.text)

        return PredictResponse(
            is_positive=bool(pred_label),  # 1 = positif, 0 = négatif
            score=float(pred_proba)
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Erreur d'inférence : {e}")  # pour debug
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/feedback")
async def post_feedback(data: dict):
    """
    Route pour collecter les retours utilisateurs.
    Si la prédiction était incorrecte, le feedback est loggué sur Azure.
    """
    if data.get("is_correct") is False:
        logger.warning("NEGATIVE_FEEDBACK", extra={
            "custom_dimensions": {
                "tweet": data.get("text"),
                "prediction": str(data.get("prediction"))
            }
        })
        return {"status": "Alerte logguée sur Azure"}

    return {"status": "Feedback positif reçu"}