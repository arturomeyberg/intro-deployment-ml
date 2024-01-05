import joblib
from pydantic import BaseModel

class predictionRequest(BaseModel):
    opening_gross: float
    screens : float
    production_budget: float
    title_year: int
    aspect_ratio: float
    duration: int
    budget: float
    imdb_score: float

class predictionResponse(BaseModel):
    worldwide_gross:float