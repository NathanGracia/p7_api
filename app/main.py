from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.inference import ModelService

# Instancie l'application FastAPI
app = FastAPI(title="Tweet Sentiment API (Scikit-Learn)", version="1.1.0")

# MISE À JOUR : On utilise le modèle Logistique et le Vectoriseur TF-IDF
model_service = ModelService(
    model_path="models/logistic_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
)

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