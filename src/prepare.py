from dvc import api
import pandas as pd
from io import StringIO
import sys
import logging
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%m:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)
logging.info('Fechting data ...')

moviePath = api.read('dataset/movies.csv',remote = 'dataset-track', encoding="utf8")
openingPath = api.read('dataset/opening_gross.csv',remote = 'dataset-track', encoding="utf8")
finantialPath = api.read('dataset/finantials.csv',remote = 'dataset-track', encoding="utf8")

finDf = pd.read_csv(StringIO(finantialPath))
movieDf = pd.read_csv(StringIO(moviePath))
openingDf = pd.read_csv(StringIO(openingPath))
numeric_columns = movieDf.select_dtypes(include=['float','int']).columns.to_numpy()
numeric_columns = np.append(numeric_columns,['movie_title'])
movieDf = movieDf[numeric_columns]
finDf = finDf[['movie_title','production_budget','worldwide_gross']]
#breakpoint()
finMovieData = pd.merge(finDf,movieDf,on='movie_title',how='left')
fullMovieData = pd.merge(openingDf, finMovieData, on= 'movie_title',how='left')
fullMovieData = fullMovieData.drop(['gross','movie_title'],axis=1)
fullMovieData.to_csv('dataset/full_data.csv',index=False)
logger.info('Data fetched and prepared')

