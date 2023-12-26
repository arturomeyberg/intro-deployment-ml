from sklearn.pipeline import Pipeline
from joblib import dump
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

def updateModel(model:  Pipeline)-> None:
    dump(model,'model/model.pkl')

def saveSimpleMetricsReport(trainScore:float,
                            testScore:float,
                            valScore:float,
                            model)->None:
    with open('report.txt','w') as reportFile:
        reportFile.write('# Pipeline description')
        for key, value in model.named_steps.items():
            reportFile.write('### {key}: {value.__repr__()}' +'\n')
        reportFile.write("## Train Score {trainScore}"+'\n')
        reportFile.write("## Test Score {testScore}"+'\n')
        reportFile.write("## Validation Score {valScore}"+'\n')

def getModelPerformanceTestSet(yReal:pd.Series,yPredicted:pd.Series)-> None:
    fig, ax= plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=yPredicted,y=yReal,ax=ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('R2 of model prediction')
    fig.savefig('predictionBehavior.png')
