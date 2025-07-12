import pandas as pd
import numpy as np
class BaseModel:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def prep_data(self,X,y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        else:
            raise TypeError("Invalid Type!")
        
    def __lr_schedule(self,t):
        t0,t1 = 5,50
        return t0/(t+t1)
    
    def predict(self,X):
        return np.dot(self.coef, X) + self.intercept