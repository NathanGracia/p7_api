from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Schéma d'entrée pour la prédiction de sentiment."""
    text: str = Field(..., description="Tweet text to analyze")


class PredictResponse(BaseModel):
    """Schéma de sortie avec le résultat et le score de confiance."""
    is_positive: bool
    score: float  # probabilité entre 0 et 1


class HealthResponse(BaseModel):
    """Schéma pour le healthcheck."""
    status: str
