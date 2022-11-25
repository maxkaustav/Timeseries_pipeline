from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from PandasPipeline import FeatureExtractor

class DataTransformPipeLine:
    def __init__(self,features,index):
        self.impute=ColumnTransformer([
    ('impute_',SimpleImputer(),features),
    ],remainder='passthrough')
        self.normalizer= trf2 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(index[0],index[1]))
    ])
        self.model=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=6)
    
    def get_pipe(self):
        return Pipeline([
    ('extractor',FeatureExtractor(-1)),
    ('imputer',self.impute),
    ('normalizer',self.normalizer),
    ('Model',self.model)
    ])

