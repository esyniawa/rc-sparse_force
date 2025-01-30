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


class SpaRCeESN(nn.Module):
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

        def reset_state(self, batch_size=1):
            """Reset reservoir state and output with proper batch dimension"""
            self.V = torch.zeros(batch_size, self.dim_reservoir).to(self.device)
            self.x = torch.zeros(batch_size, self.dim_reservoir).to(self.device)
            self.output = torch.zeros(batch_size, self.dim_output).to(self.device)

        def compute_percentile_thresholds(self, V_batch):
            """Compute the n-th percentile of absolute activity for each neuron"""
            V_abs = torch.abs(V_batch)
            self.P_n = torch.quantile(V_abs, self.percentile_n / 100, dim=0)

        def forward_V(self, input_signal: torch.Tensor):
            """
            Run reservoir forward and update state V
            Args:
                input_signal: tensor of shape (batch_size, input_dim)
            """
            batch_size = input_signal.size(0)

            # Expand recurrent weights for batch processing
            W_rec_expanded = self.W_rec.unsqueeze(0).expand(batch_size, -1, -1)
            V_expanded = self.V.unsqueeze(1)

            # Compute recurrent contribution
            rec_contrib = torch.bmm(V_expanded, W_rec_expanded).squeeze(1)

            # Compute input contribution
            in_contrib = torch.matmul(input_signal, self.W_in.t())

            # Add noise
            noise = torch.randn(batch_size, self.dim_reservoir).to(self.device) * self.noise_scaling

            # Update state
            self.V = (1 - self.alpha) * self.V + self.alpha * self.f_activation(
                rec_contrib + in_contrib + noise
            )

            return self.V

        def compute_sparse_x(self, V: torch.Tensor):
            """Compute sparse representation x using thresholds"""
            theta = self.P_n + self.theta_tilde
            V_sign = torch.sign(V)
            V_abs = torch.abs(V)
            x = V_sign * self.x_activation(V_abs - theta.unsqueeze(0))
            return x

        def forward(self, input_signal: torch.Tensor):
            """Full forward pass including sparse readout"""
            # Update reservoir state
            V = self.forward_V(input_signal)

            # Compute sparse representation
            self.x = self.compute_sparse_x(V)

            # Compute output
            self.output = torch.matmul(self.x, self.W_o.t())
            return self.output

        def update_thresholds(self, target: torch.Tensor):
            """Update thresholds using learning rule from paper

            Args:
                target: One-hot encoded target tensor (batch_size, output_dim)
            """
            # Get current output and sparse representation
            output = self.output  # shape: (batch_size, output_dim)
            x = self.x  # shape: (batch_size, reservoir_dim)
            batch_size = output.size(0)

            # For each sample in batch
            delta_theta_1 = torch.zeros_like(self.theta_tilde)  # Correlation term
            delta_theta_2 = torch.zeros_like(self.theta_tilde)  # Classification term

            for b in range(batch_size):
                # Get sign of reservoir activity for this sample
                x_sign = torch.sign(self.V[b])

                # Correlation term (equation B5) - increases thresholds for correlated neurons
                # Sum over all output classes j and reservoir neurons l
                delta_1 = 0
                for j in range(self.dim_output):
                    # Get l1 term
                    l1 = x[b] * self.W_o[j]  # element-wise multiply
                    # Get l2 term
                    l2 = x[b] * self.W_o[j]
                    delta_1 += torch.outer(l1, l2).sum(0)  # Sum over output dimension
                delta_theta_1 += delta_1 * x_sign

                # Classification term (equation B6) - decreases thresholds that help correct class
                correct_class = target[b].argmax()
                delta_theta_2 -= self.W_o[correct_class] * x_sign

            # Average over batch maybe normalization?
            delta_theta_1 = delta_theta_1 / batch_size
            delta_theta_2 = delta_theta_2 / batch_size

            # Update thresholds
            with torch.no_grad():
                self.theta_tilde += self.lr_threshold * (delta_theta_1 + delta_theta_2)