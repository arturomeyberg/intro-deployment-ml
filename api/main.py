from fastapi import FastAPI
from app.models import predictionRequest, predictionResponse
from app.views import getPrediction

app = FastAPI(docs_url="/")

@app.post('/v1/prediction')

def makeModelPrediction( request: predictionRequest):
    return predictionResponse(worldwide_gross = getPrediction(request))