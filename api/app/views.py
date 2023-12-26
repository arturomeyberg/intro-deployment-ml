from .models import predictionRequest
from .utils import getModel, transformData

model = getModel()

def getPrediction(request: predictionRequest)-> float:
    dataToPredict = transformData(request)
    prediction = model.predict(dataToPredict)[0]
    return max(0,prediction)