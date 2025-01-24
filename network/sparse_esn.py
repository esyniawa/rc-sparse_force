import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import os
from collections import deque


def initialize_reservoir_weights(dim_reservoir: int,
                                 probability_recurrent_connection: float,
                                 spectral_radius: float,
                                 normalize_eig_values: bool = True):
    """
    Initializes the reservoir weight matrix using sparse connectivity and spectral scaling
    Returns:
        torch.Tensor: Initialized weight matrix scaled to desired spectral radius
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


class SpaRCeLoss:
    def __init__(self, loss_type='mse'):
        """
        Initialize SpaRCe loss function
        Args:
            loss_type: str, either 'mse' for mean squared error or
                      'sigmoidal_cross_entropy' for classification tasks
        """
        self.loss_type = loss_type

    def __call__(self, output, target):
        if self.loss_type == 'mse':
            # Mean squared error (equation 3 in paper)
            return 0.5 * torch.mean((target - output) ** 2)

        elif self.loss_type == 'sigmoidal_cross_entropy':
            # Sigmoidal cross entropy (equation 9 in paper)
            output_sigmoid = torch.sigmoid(output)
            return -torch.mean(
                target * torch.log(output_sigmoid + 1e-10) +
                (1 - target) * torch.log(1 - output_sigmoid + 1e-10)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class SparseESN(nn.Module):
    def __init__(self,
                 dim_reservoir: int,
                 dim_input: int,
                 dim_output: int,
                 # Reservoir parameters
                 history_capacity: int = 100,
                 tau: float = 0.01,
                 dt: float = 0.001,
                 noise_scaling: float = 0.0,
                 f_activation_func: torch.nn.Module = nn.Tanh(),
                 x_activation_func: torch.nn.Module = nn.ReLU(),
                 # Connection parameters
                 spectral_radius: float = 1.0,
                 probability_recurrent_connection: float = 1.0,
                 feedforward_scaling: float = 1.0,
                 percentile_n: float = 75.0,  # n-th percentile for sparsity threshold
                 theta_tilde_init: str = 'zeros',  # 'zeros' or 'uniform'
                 # Training parameters
                 learning_rate_threshold: float = 0.01,
                 learning_rate_readout: float = 0.01,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        assert 0 <= percentile_n <= 100, "Percentile must be between 0 and 100"

        if seed is not None:
            torch.manual_seed(seed)

        # Dimensions
        self.dim_reservoir = dim_reservoir
        self.dim_input = dim_input
        self.dim_output = dim_output

        # Model parameters
        self.alpha = dt / tau  # leakage term
        self.device = device
        self.percentile_n = percentile_n
        self.f_activation = f_activation_func
        self.x_activation = x_activation_func
        self.lr_threshold = learning_rate_threshold
        self.lr_readout = learning_rate_readout

        # Initialize weights
        self.W_rec = initialize_reservoir_weights(dim_reservoir=self.dim_reservoir,
                                                  probability_recurrent_connection=probability_recurrent_connection,
                                                  spectral_radius=spectral_radius).to(device)
        self.W_in = (torch.empty(dim_reservoir, dim_input).uniform_(-1, 1) * feedforward_scaling).to(device)
        self.W_o = nn.Parameter(torch.zeros(dim_output, dim_reservoir)).to(device)

        # Initialize thresholds
        self.register_buffer('P_n', torch.zeros(dim_reservoir))  # Fixed percentile component
        self.theta_tilde = nn.Parameter(torch.zeros(dim_reservoir))  # Learnable component
        if theta_tilde_init == 'uniform':
            nn.init.uniform_(self.theta_tilde, -0.1, 0.1)

        # Initialize states
        self.noise_scaling = noise_scaling
        self.reset_state()

        # History of past states
        self.history_capacity = history_capacity
        self.V_history = deque(maxlen=history_capacity)

    def reset_state(self):
        """Reset reservoir state and output"""
        self.V = torch.zeros(self.dim_reservoir).to(self.device)
        self.x = torch.zeros(self.dim_reservoir).to(self.device)
        self.output = torch.zeros(self.dim_output).to(self.device)

    def compute_percentile_thresholds(self, V_batch):
        """Compute the n-th percentile of absolute activity for each neuron"""
        V_abs = torch.abs(V_batch)
        self.P_n = torch.quantile(V_abs, self.percentile_n / 100, dim=0)

    def forward_V(self, input_signal: torch.Tensor):
        """Run reservoir forward and update state V"""
        self.V = (1 - self.alpha) * self.V + self.alpha * self.f_activation(
            torch.matmul(self.W_rec, self.V) +
            torch.matmul(self.W_in, input_signal.float()) +
            torch.randn(self.dim_reservoir).to(self.device) * self.noise_scaling
        )
        self.V_history.append(self.V.detach())
        return self.V

    def compute_sparse_x(self, V: torch.Tensor):
        """Compute sparse representation x using thresholds"""
        theta = self.P_n + self.theta_tilde
        V_sign = torch.sign(V)
        V_abs = torch.abs(V)
        x = V_sign * self.x_activation(V_abs - theta)
        return x

    def forward(self, input_signal: torch.Tensor):
        """Full forward pass including sparse readout"""
        # Update reservoir state
        V = self.forward_V(input_signal)

        # Compute sparse representation
        self.x = self.compute_sparse_x(V)

        # Compute output
        self.output = torch.matmul(self.W_o, self.x)
        return self.output

    def update_thresholds(self, target: torch.Tensor):
        """Update thresholds using learning rule from paper"""
        # Get current output and sparse representation
        output = self.output
        x = self.x

        # Compute gradient components (equations 7-8 from paper)
        delta_theta_1 = torch.zeros_like(self.theta_tilde)
        delta_theta_2 = torch.zeros_like(self.theta_tilde)

        for j in range(self.dim_output):
            # Correlation term (equation 7)
            delta_theta_1 += output[j] * torch.sum(
                self.W_o[j].unsqueeze(0) * self.W_o.detach(), dim=0
            ) * torch.sign(x)

            # Task-specific term (equation 8)
            if target[j] == 1:  # Assuming one-hot encoded targets
                delta_theta_2 -= self.W_o[j] * torch.sign(x)

        # Update thresholds
        with torch.no_grad():
            self.theta_tilde += self.lr_threshold * (delta_theta_1 + delta_theta_2)

