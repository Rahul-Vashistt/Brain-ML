import pandas as pd
import numpy as np

class BaseModel:
    """
    A base model class providing shared functionality across ML algorithms.

    Handles:
    - Optimizer strategy
    - Learning hyperparameters
    - Intercept handling
    - Feature scaling (Standard/MinMax)
    - Data preparation (fit + predict)
    """

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
        """
        Initialize the base model with core hyperparameters and preprocessing settings.
        
        Parameters
        ----------
        method : str
            Optimization method (e.g., 'ols', 'sgd', 'batch_gd', 'mbgd').

        learning_rate : float
            Learning rate for gradient-based optimizers.

        epochs : int
            Number of training iterations.

        batch_size : int
            Batch size for mini-batch gradient descent.

        n_batches : int
            Number of batches to be used in mini-batch gradient descent.

        tol : float
            Convergence tolerance for stopping gradient descent.

        fit_intercept : bool
            Whether to add an intercept term to features.

        normalize : str or None
            Scaling strategy to apply. Supported values: 'standard', 'minmax', or None.
        """
        self.coef = None
        self.intercept = None
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize.lower() if normalize else None  # Make case-insensitive
        self.scaler = None  # Will hold the fitted scaler
        self.scaler_used = True  # Flag to warn on double scaling

    def transform_X(self, X, training=False):
        """
        Apply preprocessing to the feature matrix X.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.

        training : bool
            Whether this transformation is happening during model training (fit).
            If False, the scaler must already be fitted (used in predict).
        
        Returns
        -------
        X : np.ndarray
            Transformed (and possibly scaled) feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.normalize is not None:
            from Brain.preprocessing import StandardScaler, MinMaxScaler
            from Brain.utils import is_standardized, is_minmax_scaled

            if training:
                # Fitting the scaler during training
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

            else:
                # Use previously fitted scaler during prediction
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. You must call `fit()` before `predict()`.")
                X = self.scaler.transform(X)

        # Optionally add intercept term (bias column of ones)
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        return X

    def prep_data(self, X, y):
        """
        Prepares feature matrix and target for training.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.

        y : array-like or pd.Series
            Target vector.

        Returns
        -------
        X : np.ndarray
            Processed feature matrix.
        y : np.ndarray
            Target values as NumPy array.
        """
        X = self.transform_X(X, training=True)

        if isinstance(y, pd.Series):
            y = y.values

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        return X, y

    def predict(self, X):
        """
        Predict target values using the learned model parameters.

        Parameters
        ----------
        X : array-like
            Input features for prediction.

        Returns
        -------
        y_pred : np.ndarray
            Predicted target values.
        """
        X = self.transform_X(X, training=False)
        return np.dot(X, self.coef)

    @property
    def coef_(self):
        """Returns learned coefficients after training."""
        return self.coef

    @property
    def intercept_(self):
        """Returns learned intercept after training."""
        return self.intercept

    def summary(self):
        """Prints a summary of the fitted model."""
        print("Method:", self.method)
        print("Intercept:", self.intercept_)
        print("Coefficients:", self.coef_)
        print("Scaler used:", self.scaler.__class__.__name__ if self.scaler else "None")
        print("Fitted:", self.coef is not None)
