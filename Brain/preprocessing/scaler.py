import pandas as pd
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self,X):
        # Compute the mean and std to be used for later scaling.
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_==0] = 1
        return self

    def transform(self,X):
        # Scale features of X.
        return (X - self.mean_) / self.std_
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self,X_scaled):
        # Scale back to original values.
        return X_scaled * self.std_ + self.mean_
    

class MinMaxScaler:
    def __init__(self,feature_range=(0,1)):
        self.min_ = None
        self.max_ = None
        self.data_min = feature_range[0]
        self.data_max = feature_range[1]

    def fit(self,X):
        self.max_ = np.max(X, axis=0)
        self.min_ = np.min(X, axis=0)
        return self
    
    def transform(self,X):
        scaled = (X - self.min_) / (self.max_ - self.min_)
        return scaled * (self.data_max - self.data_min) + self.data_min
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self,X_scaled):
        scaled = (X_scaled - self.data_min) / (self.data_max - self.data_min)
        return scaled * (self.max_ - self.min_) + self.min_