from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self,shift_:int) -> None:
        self.shift_=shift_
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X["lag_sales_1"] = X.groupby("Product_Code")['Sales'].shift(1)
        X["diff_sales_1"] = X.groupby("Product_Code")['Sales'].diff(1)
        X["mean_sales_4"] = X.groupby("Product_Code")['Sales'].rolling(4).mean().reset_index(level=0, drop=True)

        return X