import pandas as pd
import numpy as np
from Brain.core import BaseModel
from Brain.utils import resolve_optimizer
from Brain.optimization import Batch_GD, Stochastic_GD, Mini_Batch_GD
from Brain.method import optimizers

class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self,X,y):
        """
    Fit linear regression model.

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target

    Returns
    -------
    self : object
        Returns self.
    """
        X,y = self.prep_data(X,y)
        method = resolve_optimizer(self.method)

        if method == 'ols':
            theta = np.pinv(X.T @ X) @ X.T @ y
        elif method == 'batch_gd':
            theta = Batch_GD(X, y, self.learning_rate, self.epochs, self.tol)
        elif method == 'sgd':
            theta = Stochastic_GD(X, y, self.learning_rate, self.epochs, self.tol)
        elif method == 'mbgd':
            theta = Mini_Batch_GD(X, y, self.learning_rate, self.epochs, self.batch_size, self.n_batches, self.tol)

        self.coef = theta[1:]
        self.intercept = theta[0]
        return self