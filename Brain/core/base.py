import pandas as pd
import numpy as np

class BaseModel:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def prep_data(self,X,y):
        if isinstance(X, pd.DataFrame):
            X = X.values
            X = np.insert(X,0,1,axis=1)
        if isinstance(y, pd.Series):
            y = y.values
        else:
            raise TypeError("Invalid Type!")
        return X,y
    
    def predict(self,X):
        return np.dot(self.coef, X) + self.intercept
    
    @property
    def coef_(self):
        return self.coef
    
    @property
    def intercept_(self):
        return self.intercept