import numpy as np


class Metrics:
    def __init__(self) -> None:
        pass
    
    def mape(y_true, y_pred):
        ape = np.abs((y_true - y_pred) / y_true)
        #ape[~np.isfinite(ape)] = 0. # VERY questionable
        ape[~np.isfinite(ape)] = 1. # pessimist estimate
        return np.mean(ape)
        
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))