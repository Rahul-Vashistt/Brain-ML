import pandas as pd
import numpy as np

class BaseModel:
    def __init__(
        self, 
        method=None, 
        learning_rate=None, 
        epochs=1000, 
        batch_size=None, 
        n_batches=None, 
        tol=1e-06, 
        fit_intercept: bool=True, 
        normalize=None
    ):
        self.coef = None
        self.intercept = None
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize.lower() if normalize else None
        self.scaler = None
        self.scaler_used = True

    def transform_X(self, X, training=False):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.normalize is not None:
            from Brain.preprocessing import StandardScaler, MinMaxScaler
            from Brain.utils import is_standardized, is_minmax_scaled

            if training:
                if self.normalize in ['standard', 'std']:
                    if self.scaler_used and is_standardized(X):
                        print("⚠️ Data may already be standardized. Double-scaling could distort model performance.")
                    self.scaler = StandardScaler()
                    X = self.scaler.fit_transform(X)

                elif self.normalize in ['minmax', 'min_max']:
                    if self.scaler_used and is_minmax_scaled(X):
                        print("⚠️ Data may already be min-max scaled. Double-scaling could distort model performance.")
                    self.scaler = MinMaxScaler()
                    X = self.scaler.fit_transform(X)
                else:
                    raise ValueError("normalize must be one of ['standard', 'minmax', None]")

                self.scaler_used = True

            else:  # during prediction
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. You must call `fit()` before `predict()`.")
                X = self.scaler.transform(X)

        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        return X

    def prep_data(self, X, y):
        X = self.transform_X(X, training=True)

        if isinstance(y, pd.Series):
            y = y.values

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        return X, y

    def predict(self, X):
        X = self.transform_X(X, training=False)
        return np.dot(X, self.coef)

    @property
    def coef_(self):
        return self.coef

    @property
    def intercept_(self):
        return self.intercept

    def summary(self):
        print("Method:", self.method)
        print("Intercept:", self.intercept_)
        print("Coefficients:", self.coef_)
        print("Scaler used:", self.scaler.__class__.__name__ if self.scaler else "None")
        print("Fitted:", self.coef is not None)
