import numpy as np

class OrdinaryLeastSquaresRegression:
    """
    Simple Ordinary Least Squares Regression implementation
    """
    
    def __init__(self):
        self.coefs_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the OLS model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Add intercept column (column of ones)
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        # Calculate coefficients using the normal equation
        # β = (X^T X)^(-1) X^T y
        self.coefs_with_intercept_ = np.linalg.lstsq(
            X_with_intercept.T @ X_with_intercept,
            X_with_intercept.T @ y,
            rcond=None
        )[0]
        
        # Separate intercept and coefficients
        self.intercept_ = self.coefs_with_intercept_[0]
        self.coefs_ = self.coefs_with_intercept_[1:]
        
        return self
    
    def predict(self, X):
        """
        Predict using the OLS model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if self.coefs_ is None:
            raise ValueError("Model has not been fitted yet!")
        
        return X @ self.coefs_ + self.intercept_