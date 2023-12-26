from joblib import load 
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
import pandas as pd
import os
from io import BytesIO

def getModel()-> Pipeline:
    modelPath = os.environ.get('MODEL_PATH','../model/model.pkl')
    
    model = load(modelPath)
    return model

def transformData(classModel: BaseModel) -> pd.DataFrame:
    transitionDict = {key: [value] for key, value in classModel.dict().items() }
    dataFrame = pd.DataFrame(transitionDict)
    return dataFrame