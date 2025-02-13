from sklearn.preprocessing import StandardScaler
import numpy as np


class CustomScaler(StandardScaler):
    def __init__(self, target_variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.target_variance = target_variance

    def transform(self, X):
        X_scaled = super().transform(X)
        return X_scaled * np.sqrt(self.target_variance)

    def inverse_transform(self, X):
        # First undo the variance scaling
        X_unscaled = X / np.sqrt(self.target_variance)
        # Then use parent class inverse_transform
        return super().inverse_transform(X_unscaled)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


"""
# Example usage
scaler = CustomScaler(target_variance=0.1)

X = np.array([[1, 2], [3, 4], [5, 6]])
X_scaled = scaler.fit_transform(X)
print(X_scaled)
print(scaler.inverse_transform(X_scaled))
"""
