from pydantic import BaseModel, Field

#  définit le schéma d'entrée
class PredictRequest(BaseModel):
    text: str = Field(..., description="Tweet text to analyze")

#  définit le schéma de sortie
class PredictResponse(BaseModel):
    is_positive: bool
    score: float  #  renvoie une proba (0..1)

#  définit un schéma pour le healthcheck
class HealthResponse(BaseModel):
    status: str
