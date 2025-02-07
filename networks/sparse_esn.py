import torch
import torch.nn as nn
from typing import Optional
from .utils import initialize_reservoir_weights, z_normalize


def mse_loss(output: torch.Tensor, target: torch.Tensor):
    return 0.5 * torch.mean((target - output) ** 2)


class SpaRCeLoss:
    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize SpaRCe loss function
        :param loss_type: either 'mse' for mean squared error or 'sigmoidal_cross_entropy' for classification tasks
        """
        self.loss_type = loss_type

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        if self.loss_type == 'mse':
            # Mean squared error (equation 3 in paper)
            # They also mentioned Ridge regression as well
            return mse_loss(output, target)

        elif self.loss_type == 'sigmoidal_cross_entropy':
            # Sigmoidal cross entropy (equation 9 in paper)
            output_sigmoid = torch.sigmoid(output)
            return -torch.mean(
                target * torch.log(output_sigmoid + 1e-10) +
                (1 - target) * torch.log(1 - output_sigmoid + 1e-10)
            )

        elif self.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()(output, target)

        elif self.loss_type == 'bce':
            return nn.BCELoss()(output, target)

        elif self.loss_type == 'td_error':
            pass

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class SpaRCeESN(nn.Module):
        def __init__(self,
                     dim_reservoir: int,
                     dim_input: int,
                     dim_output: int,
                     mode: str,
                     # Reservoir parameters
                     alpha: float = 0.1,  # leakage term
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
                     learning_rate_threshold: float = 0.001,  # Should be lower than learning_rate_readout
                     learning_rate_readout: float = 0.01,
                     seed: Optional[int] = None,
                     device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
            super().__init__()

            assert mode in ('regression', 'classification'), "Mode must be either 'regression' or 'classification'"
            assert 0 <= percentile_n <= 100, "Percentile must be between 0 and 100"

            if seed is not None:
                torch.manual_seed(seed)

            self.mode = mode

            # Dimensions
            self.dim_reservoir = dim_reservoir
            self.dim_input = dim_input
            self.dim_output = dim_output

            # Model parameters
            self.alpha = alpha
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
            self.W_o = nn.Parameter(torch.zeros(dim_output, dim_reservoir)).to(device)  # Learnable output weights

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
            """Update thresholds using learning rule from paper with Z normalization. TODO: Maybe loss clipping?"""

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

                # Correlation term (equation B5)
                # For each output dimension j
                for j in range(self.dim_output):
                    # W_jl x_l term - weighted outputs for all neurons
                    weighted_outputs = x[b] * self.W_o[j]  # shape: (dim_reservoir,)

                    # W_jk sign(x_k) term - sign-weighted contribution
                    sign_weighted = self.W_o[j] * x_sign  # shape: (dim_reservoir,)

                    # Outer product to get correlation matrix
                    delta_1 = torch.outer(weighted_outputs, sign_weighted).diagonal()

                    # Sum over output dimension j
                    delta_theta_1 += delta_1

                # Classification term (equation B6)
                if self.mode == 'classification':
                    correct_class = target[b].argmax()
                    delta_theta_2 -= self.W_o[correct_class] * x_sign

            # Apply Z normalization to each delta term
            delta_theta_1 = z_normalize(delta_theta_1)
            if self.mode == 'classification':
                delta_theta_2 = z_normalize(delta_theta_2)

            # Update thresholds
            with torch.no_grad():
                self.theta_tilde += self.lr_threshold * (delta_theta_1 + delta_theta_2)

        def save_model(self, path: str):
            torch.save(self.state_dict(), path)

        def load_model(self, path: str):
            self.load_state_dict(torch.load(path))
