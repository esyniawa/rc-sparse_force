import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle


class VarianceScaler(StandardScaler):
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


class ZeroPreservingScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that preserves zero values exactly and scales other
    values to a specified range, treating positive and negative values separately.

    :param feature_range: Desired range of transformed data. default=(-1, 1)
    :param clip: Whether to clip transformed values to the specified feature range.
    :param epsilon : Small value to avoid division by zero.
    """
    def __init__(self, feature_range=(-1, 1), clip=True, epsilon=1e-10):
        self.feature_range = feature_range
        self.clip = clip
        self.epsilon = epsilon
        self.min_ = None
        self.max_ = None
        self.scale_pos_ = None
        self.scale_neg_ = None
        self.min_bound_ = feature_range[0]
        self.max_bound_ = feature_range[1]

    def fit(self, X, y=None):
        """
        Compute the scaling factors to be used for scaling.
        """
        X = np.asarray(X)

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Store original min and max
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        # Calculate separate scaling factors for positive and negative values
        X_pos = X.copy()
        X_pos[X_pos < 0] = 0
        X_neg = X.copy()
        X_neg[X_neg > 0] = 0

        # Maximum absolute values in positive and negative domains
        pos_max = np.max(X_pos, axis=0)
        neg_max = np.abs(np.min(X_neg, axis=0))

        # Replace zeros with epsilon to avoid division by zero
        pos_max[pos_max < self.epsilon] = 1.0
        neg_max[neg_max < self.epsilon] = 1.0

        # Compute scaling factors
        self.scale_pos_ = self.max_bound_ / pos_max
        self.scale_neg_ = abs(self.min_bound_) / neg_max

        return self

    def transform(self, X):
        X = np.asarray(X)

        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        check_is_fitted = hasattr(self, 'scale_pos_') and hasattr(self, 'scale_neg_')
        if not check_is_fitted:
            raise ValueError("This ZeroPreservingScaler instance is not fitted yet. "
                             "Call 'fit' before using this estimator.")

        # Apply different scaling to positive and negative values
        X_transformed = np.zeros_like(X, dtype=float)

        # For each feature dimension
        for j in range(X.shape[1]):
            # Scale positive values
            mask_pos = X[:, j] > 0
            X_transformed[mask_pos, j] = X[mask_pos, j] * self.scale_pos_[j]

            # Scale negative values - multiply by scaling factor but KEEP the negative sign
            mask_neg = X[:, j] < 0
            # This preserves the negative sign while applying the scaling
            X_transformed[mask_neg, j] = -1.0 * abs(X[mask_neg, j]) * self.scale_neg_[j]

        # Clip values to feature_range if requested
        if self.clip:
            X_transformed = np.clip(X_transformed, self.min_bound_, self.max_bound_)

        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            X_transformed = X_transformed.ravel()

        return X_transformed

    def inverse_transform(self, X):
        """ Undo the scaling. """
        X = np.asarray(X)

        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        check_is_fitted = hasattr(self, 'scale_pos_') and hasattr(self, 'scale_neg_')
        if not check_is_fitted:
            raise ValueError("This ZeroPreservingScaler instance is not fitted yet. "
                             "Call 'fit' before using this estimator.")

        # Apply inverse scaling to positive and negative values separately
        X_orig = np.zeros_like(X, dtype=float)

        # For each feature dimension
        for j in range(X.shape[1]):
            # Inverse scale positive values
            mask_pos = X[:, j] > 0
            X_orig[mask_pos, j] = X[mask_pos, j] / self.scale_pos_[j]

            # Inverse scale negative values
            mask_neg = X[:, j] < 0
            # Preserve the negative sign during inverse transform
            X_orig[mask_neg, j] = -1.0 * abs(X[mask_neg, j]) / self.scale_neg_[j]

        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            X_orig = X_orig.ravel()

        return X_orig

    def save(self, filename):
        """Save the scaler to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load a saved scaler from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


# Example usage
if __name__ == "__main__":
    # Generate sample data with positive and negative values
    np.random.seed(42)
    data = np.random.randn(100, 2) * 10  # Random data around zero with different scales
    data[::3] = 0  # Set some values to exactly zero

    # Add asymmetric data to test
    data[:50, 0] = np.abs(data[:50, 0]) * 2  # Higher positive values for feature 0
    data[50:, 1] = -np.abs(data[50:, 1]) * 3  # Higher negative values for feature 1

    # Create and fit the scaler
    scaler = ZeroPreservingScaler(feature_range=(-1, 1))
    scaler.fit(data)

    # Transform the data
    scaled_data = scaler.transform(data)

    # Verify zero preservation
    zero_indices = np.where(np.abs(data) < 1e-10)
    zero_preservation = np.allclose(scaled_data[zero_indices], 0)
    print(f"Zero values preserved: {zero_preservation}")

    # Verify range
    print(f"Scaled data min: {scaled_data.min(axis=0)}")
    print(f"Scaled data max: {scaled_data.max(axis=0)}")

    # Test with 1D data
    data_1d = np.random.randn(50) * 10
    data_1d[::5] = 0  # Set some values to exactly zero

    scaler_1d = ZeroPreservingScaler(feature_range=(-1, 1))
    scaler_1d.fit(data_1d)
    scaled_1d = scaler_1d.transform(data_1d)

    print("\n1D data test:")
    print(f"Input shape: {data_1d.shape}, Output shape: {scaled_1d.shape}")
    zero_indices_1d = np.where(np.abs(data_1d) < 1e-10)[0]
    if len(zero_indices_1d) > 0:
        print(f"Zero values preserved in 1D: {np.allclose(scaled_1d[zero_indices_1d], 0)}")

    print(f"1D scaled data min: {scaled_1d.min()}")
    print(f"1D scaled data max: {scaled_1d.max()}")

    # Verify inverse transform
    recovered_data = scaler.inverse_transform(scaled_data)
    recovery_accuracy = np.allclose(data, recovered_data)
    print(f"\nPerfect recovery: {recovery_accuracy}")
    print(f"Max absolute error: {np.abs(data - recovered_data).max()}")

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 0].set_title("Original Data")

    # Plot scaled data
    axes[0, 1].scatter(scaled_data[:, 0], scaled_data[:, 1], alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 1].set_title("Scaled Data")
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-1.1, 1.1)

    # Plot feature 0 comparison
    axes[1, 0].scatter(data[:, 0], scaled_data[:, 0], alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 0].set_title("Feature 0: Original vs Scaled")
    axes[1, 0].set_xlabel("Original")
    axes[1, 0].set_ylabel("Scaled")
    axes[1, 0].set_ylim(-1.1, 1.1)

    # Plot feature 1 comparison
    axes[1, 1].scatter(data[:, 1], scaled_data[:, 1], alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='r', linestyle='-', alpha=0.3)
    axes[1, 1].set_title("Feature 1: Original vs Scaled")
    axes[1, 1].set_xlabel("Original")
    axes[1, 1].set_ylabel("Scaled")
    axes[1, 1].set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig("zero_preserving_scaler_demo.png")
    plt.show()

    # Additional plot to better visualize the scaling function
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate a range of sample values
    x_values = np.linspace(-20, 20, 1000)

    # Create a simple 1D scaler for demonstration
    simple_scaler = ZeroPreservingScaler(feature_range=(-1, 1))
    simple_scaler.fit(x_values.reshape(-1, 1))

    # Apply transformation
    y_values = simple_scaler.transform(x_values.reshape(-1, 1)).ravel()

    # Plot the scaling function
    ax1.plot(x_values, y_values)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Scaling Function")
    ax1.set_xlabel("Original Value")
    ax1.set_ylabel("Scaled Value")
    ax1.set_ylim(-1.1, 1.1)

    # Zoom in near zero
    zoom_range = np.linspace(-5, 5, 1000)
    zoom_scaled = simple_scaler.transform(zoom_range.reshape(-1, 1)).ravel()

    ax2.plot(zoom_range, zoom_scaled)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Scaling Function (Zoomed)")
    ax2.set_xlabel("Original Value")
    ax2.set_ylabel("Scaled Value")
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig("zero_preserving_scaler_function.png")
    plt.show()
