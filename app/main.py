from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.inference import ModelService

# Instancie l'application FastAPI
app = FastAPI(title="Tweet Sentiment API", version="1.0.0")

# FIX : On pointe vers les nouveaux fichiers Word2Vec
model_service = ModelService(
    model_path="models/model_lstm_w2v.h5",
    tokenizer_path="models/tokenizer.pickle"
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        # Ton ModelService renvoie maintenant (pred, proba) via le LSTM
        pred_label, pred_proba = model_service.predict(payload.text)

        return PredictResponse(
            is_positive=bool(pred_label),  # 1 -> True, 0 -> False
            score=float(pred_proba)  # La probabilité sigmoid
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # En phase de test, tu peux décommenter la ligne suivante pour debugger
        # print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail="Inference failed")