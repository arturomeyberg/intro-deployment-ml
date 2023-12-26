from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd
from utils import updateModel, saveSimpleMetricsReport,getModelPerformanceTestSet

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%m:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)
logging.info('Loading data ...')
data = pd.read_csv('dataset/full_data.csv')
logging.info('Loading model')

#data = data.fillna(0)

model = Pipeline(
    [('imputer',SimpleImputer(strategy='mean',missing_values=np.nan))
    ,('core_model',GradientBoostingRegressor())
    ]
    )
logger.info('Separating dataset into train and test')
x= data.drop(['worldwide_gross'],axis=1)
y = data['worldwide_gross']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35,random_state=42)

logger.info('Setting hyperparameters for tunning')

param_tunning = { 'core_model__n_estimators': range(20,301,20)

}

grid_search = GridSearchCV(model, param_grid=param_tunning, scoring='r2',cv=5)
logger.info("Starting grid search")
grid_search.fit(x_train,y_train)

logger.info("Cross validation with best model")
final_result = cross_validate(grid_search.best_estimator_
                              ,x_train
                              ,y_train
                              , return_train_score=True
                              ,cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])


assert train_score > 0.7
assert test_score > 0.65

logger.info("Best train score: "+ str(train_score))
logger.info("Best test score: "+ str(test_score))

logging.info('Updating model .....')
updateModel(grid_search.best_estimator_)

logging.info('Generating model report ...')

validationScore = grid_search.best_estimator_.score(x_test,y_test)
saveSimpleMetricsReport(train_score, test_score, validationScore, model)

yTestPred = grid_search.best_estimator_.predict(x_test)
getModelPerformanceTestSet(y_test,yTestPred)

logging.info('Trainning finish ...')
