import pandas as pd
import numpy as np
from pipeline import DataTransformPipeLine
from metrics import Metrics
data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
data= data.filter(regex=r'Product|W')
melt = data.melt(id_vars='Product_Code', var_name='Week', value_name='Sales')
melt['Product_Code'] = melt['Product_Code'].str.extract('(\d+)', expand=False).astype(int)
melt['Week'] = melt['Week'].str.extract('(\d+)', expand=False).astype(int)

melt = melt.sort_values(['Week', 'Product_Code'])
valid_split_point=40
melt_train = melt[melt['Week'] < valid_split_point].copy()
melt_valid = melt[melt['Week'] >= valid_split_point].copy()

melt_train['sales_next_week'] = melt_train.groupby("Product_Code")['Sales'].shift(-1)
melt_valid['sales_next_week'] = melt_valid.groupby("Product_Code")['Sales'].shift(-1)
melt_train = melt_train.dropna()
melt_valid = melt_valid.dropna()


features = ['Sales', 'lag_sales_1', 'diff_sales_1', 'mean_sales_4']
ytr = melt_train['sales_next_week']
yval=melt_valid['sales_next_week']
xtr=melt_train.drop(['sales_next_week'],axis=1)
xval=melt_valid.drop(['sales_next_week'],axis=1)
pipeobj=DataTransformPipeLine(features=features,index=[0,7])
pipe=pipeobj.get_pipe()

pipe.fit(xtr,ytr)


print(Metrics.wmape(pipe.predict(xval),yval))