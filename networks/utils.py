import torch


def z_normalize(x: torch.Tensor, eps: float = 1e-8):
    return (x - x.mean()) / (x.std() + eps)


def initialize_reservoir_weights(dim_reservoir: int,
                                 probability_recurrent_connection: float,
                                 spectral_radius: float,
                                 normalize_eig_values: bool = True):
    """
    Initializes the reservoir weight matrix using sparse connectivity and spectral scaling
    :return Initialized weight matrix scaled to desired spectral radius
    """

    # Create sparse mask using bernoulli distribution
    mask = (torch.rand(dim_reservoir, dim_reservoir) < probability_recurrent_connection).float()

    # Initialize weights using normal distribution
    weights = torch.randn(dim_reservoir, dim_reservoir) * \
              torch.sqrt(torch.tensor(1.0 / (probability_recurrent_connection * dim_reservoir)))

    # Apply mask to create sparse connectivity
    W = mask * weights

    # Scale matrix to desired spectral radius
    if normalize_eig_values:
        eigenvalues = torch.linalg.eigvals(W)
        max_abs_eigenvalue = torch.max(torch.abs(eigenvalues))
        W /= max_abs_eigenvalue

    return W * spectral_radius
