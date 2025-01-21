import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import os


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


class Reservoir(nn.Module):
    def __init__(self,
                 dim_reservoir: int,
                 dim_input: int,
                 dim_output: int,
                 tau: float = 10.0,
                 chaos_factor: float = 1.5,
                 probability_recurrent_connection: float = 0.1,
                 feedforward_scaling: float = 1.0,
                 feedback_scaling: float = 1.0,
                 w_out_initialization: Optional[str] = None,  # None means zeros else 'uniform' or 'normal'
                 noise_scaling: float = 0.05,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # Model parameters
        self.dim_reservoir = dim_reservoir
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.tau = tau
        self.device = device

        # Move model to specified device
        self.to(device)

        # Initialize weights
        self.W_rec = initialize_reservoir_weights(dim_reservoir=self.dim_reservoir,
                                                  probability_recurrent_connection=probability_recurrent_connection,
                                                  spectral_radius=chaos_factor).to(device)
        self.W_in = (torch.empty(dim_reservoir, dim_input).uniform_(-1, 1) * feedforward_scaling).to(device)
        self.W_fb = (torch.empty(dim_reservoir, dim_output).uniform_(-1, 1) * feedback_scaling).to(device)
        self.feedback_scaling = feedback_scaling

        # initialize readout weights
        if w_out_initialization == 'uniform':
            self.W_out = torch.empty(dim_output, dim_reservoir).uniform_(-1, 1).to(device)
        elif w_out_initialization == 'normal':
            self.W_out = (torch.randn(dim_output, dim_reservoir) / np.sqrt(dim_input)).to(device)
        else:
            self.W_out = torch.zeros(dim_output, dim_reservoir).to(device)

        # Initialize states
        self.noise_scaling = noise_scaling
        self.reset_state()

    @torch.no_grad()
    def forward(self, input_signal, dt: float = 0.1, compute_output: bool = True):
        # Ensure input is on correct device and properly shaped
        input_signal = input_signal.to(self.device).view(self.dim_input)

        r_tanh = torch.tanh(self.r)

        # Compute total input to reservoir neurons
        state = (torch.matmul(self.W_rec, r_tanh) +
                 torch.matmul(self.W_in, input_signal.float()) +
                 torch.randn(self.dim_reservoir).to(self.device) * self.noise_scaling)

        if self.feedback_scaling:
            state += torch.matmul(self.W_fb, self.output.float())

        # Update reservoir state
        self.r += (-self.r + state) * dt / self.tau

        # Compute output
        if compute_output:
            self.output = self.step()
            return self.output, self.r
        else:
            return self.r

    @torch.no_grad()
    def step(self):
        return torch.matmul(self.W_out, self.r)

    def reset_state(self):
        self.r = torch.zeros(self.dim_reservoir).to(self.device)
        self.output = torch.zeros(self.dim_output).to(self.device)

    def save(self, path):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ForceTrainer:
    def __init__(self,
                 reservoir: Reservoir,
                 alpha: float = 1.0):
        self.reservoir = reservoir
        self.P = torch.eye(reservoir.dim_reservoir).to(reservoir.device) / alpha

    @torch.no_grad()
    def train_step(self,
                   input_signal: torch.Tensor | np.ndarray,
                   target: torch.Tensor | np.ndarray,
                   dt: float = 0.1,
                   w_update: bool = True,
                   ):

        if isinstance(input_signal, np.ndarray):
            input_signal = torch.from_numpy(input_signal)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        # Ensure input and target are on correct device
        input_signal = input_signal.to(self.reservoir.device)
        target = target.to(self.reservoir.device)

        # Run reservoir forward
        output, r = self.reservoir.forward(input_signal, dt)

        # Compute error
        error_minus = output - target

        if w_update:
            # Update P matrix
            Pr = torch.matmul(self.P, r)
            rPr = torch.matmul(r, Pr)
            c = 1.0 / (1.0 + rPr)
            self.P -= c * torch.outer(Pr, Pr)

            # Update output weights
            self.reservoir.W_out -= c * torch.outer(error_minus, Pr)

            return error_minus
        else:
            return error_minus


class FullForceTrainer:
    def __init__(self,
                 task_network: Reservoir,
                 alpha: float = 1.0,
                 seed: Optional[int] = None,
                 clone_input_weights: bool = True,  # clones input weights from task network to the target network
                 clone_target_weights: bool = False,
                 set_recurrent_weights_to_zeros: bool = True,
                 # In the implementation of the paper, they set the recurrent weights of the task network to zero,
                 # also to asure no chaotic reservoir
                 ):
        self.task_network = task_network
        self.device = task_network.device

        # Create target network
        self.target_network = Reservoir(
            dim_reservoir=task_network.dim_reservoir,
            dim_input=task_network.dim_input + task_network.dim_output,
            # Input + target output dims (both will be given to the network)
            dim_output=task_network.dim_output,
            tau=task_network.tau,
            chaos_factor=1.0,  # No chaotic reservoir due to no feedback
            probability_recurrent_connection=1.0,
            feedback_scaling=0.0,  # No feedback in either network
            seed=seed,
            device=self.device
        )

        # Disable feedback
        self.task_network.W_fb.zero_()
        self.task_network.feedback_scaling = 0.0
        self.target_network.W_fb.zero_()

        # Some initialisations from the authors
        if clone_input_weights:
            self.target_network.W_in[:, :self.task_network.dim_input] = self.task_network.W_in[:,
                                                                        :self.task_network.dim_input]

        if clone_target_weights:
            self.target_network.W_in[:, self.task_network.dim_input:] = self.task_network.W_in[:,
                                                                        :self.task_network.dim_output]

        if set_recurrent_weights_to_zeros:
            self.task_network.W_rec.zero_()

        # Initialize single P matrix for RLS
        self.P = torch.eye(task_network.dim_reservoir).to(self.device) / alpha
        self.W_in_target = self.target_network.W_in[:, self.task_network.dim_input:]

    @torch.no_grad()
    def train_step(self,
                   input_signal: torch.Tensor,
                   target: torch.Tensor,
                   dt: float = 0.1,
                   w_update: bool = True,  # Maybe you don't want to update the weights in every iteration
                   ):
        # Ensure inputs are on correct device
        input_signal = input_signal.to(self.device)
        target = target.to(self.device)

        # Run networks forward
        rd = self.target_network.forward(torch.cat([input_signal, target]), dt,
                                         compute_output=False)  # Output not needed for training
        output, r = self.task_network.forward(input_signal, dt)

        if w_update:
            # Compute errors
            J_err = (torch.matmul(self.task_network.W_rec, r) -
                     torch.matmul(self.target_network.W_rec, rd) -
                     torch.matmul(self.W_in_target, target))

            error_minus = torch.matmul(self.task_network.W_out, r) - target

            # Update P matrix and compute gain properly
            Pr = torch.matmul(self.P, r)
            rPr = torch.matmul(r, Pr)
            k = Pr / (1 + rPr)  # with gain
            self.P -= torch.outer(Pr, k)

            # Update weights
            self.task_network.W_rec -= torch.outer(J_err, k)
            self.task_network.W_out -= torch.outer(error_minus, k)

            return error_minus

        else:
            error_minus = torch.matmul(self.task_network.W_out, torch.tanh(self.task_network.r)) - target
            return error_minus

    def reset_states(self):
        self.task_network.reset_state()
        self.target_network.reset_state()


class SparseReservoir(nn.Module):
    def __init__(self,
                 dim_reservoir: int,
                 dim_input: int,
                 dim_output: int,
                 tau: float = 10.0,
                 chaos_factor: float = 1.0,
                 probability_recurrent_connection: float = 0.1,
                 feedforward_scaling: float = 1.0,
                 noise_scaling: float = 0.05,
                 percentile_n: float = 75.0,  # n-th percentile for sparsity threshold
                 seed: int = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.dim_reservoir = dim_reservoir
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.tau = tau
        self.device = device
        self.percentile_n = percentile_n

        # Initialize weights
        self.W_rec = initialize_reservoir_weights(dim_reservoir=self.dim_reservoir,
                                                  probability_recurrent_connection=probability_recurrent_connection,
                                                  spectral_radius=chaos_factor).to(device)
        self.W_in = (torch.empty(dim_reservoir, dim_input).uniform_(-1, 1) * feedforward_scaling).to(device)
        self.W_out = torch.zeros(dim_output, dim_reservoir).to(device)

        # Initialize sparsity thresholds (theta_k) for each reservoir-output pair
        self.theta_tilde = torch.zeros(dim_reservoir).to(device)  # Learnable component

        # Initialize states
        self.noise_scaling = noise_scaling
        self.reset_state()

    @torch.no_grad()
    def compute_s_k(self, x):
        """Compute s_k according to SpaRCe paper equation:
        s_k = sign(x_k) * H(|x_k| - θ_k)
        where θ_k = P_n(|x_k|) + θ̃_k^tilde
        """
        # Absolute activities
        abs_x = torch.abs(x)

        # Total threshold is percentile + learnable component
        theta = torch.quantile(abs_x, self.percentile_n / 100) + self.theta_tilde

        # Compute s_k using sign function and tanh
        s_k = torch.sign(x) * torch.nn.functional.relu(abs_x - theta)
        return s_k

    @torch.no_grad()
    def forward(self, input_signal, dt: float = 0.1, compute_output: bool = True):
        input_signal = input_signal.to(self.device).view(self.dim_input)

        # Compute sparse activities
        s_k = self.compute_s_k(self.r)

        # Compute total input to reservoir neurons
        state = (torch.matmul(self.W_rec, s_k) +
                 torch.matmul(self.W_in, input_signal.float()) +
                 torch.randn(self.dim_reservoir).to(self.device) * self.noise_scaling)

        # Update reservoir state
        self.r += (-self.r + state) * dt / self.tau

        if compute_output:
            # Compute output using sparse activities
            self.output = torch.matmul(self.W_out, s_k)
            return self.output, s_k
        else:
            return s_k

    def reset_state(self):
        self.r = torch.zeros(self.dim_reservoir).to(self.device)
        self.output = torch.zeros(self.dim_output).to(self.device)


class SparseFullForceTrainer:
    def __init__(self,
                 task_network: SparseReservoir,
                 alpha: float = 1.0,
                 clone_input_weights: bool = True,
                 clone_target_weights: bool = False,
                 set_recurrent_weights_to_zeros: bool = True, ):

        self.task_network = task_network
        self.device = task_network.device
        self.P = torch.eye(task_network.dim_reservoir).to(self.device) / alpha

        # Create target network
        self.target_network = SparseReservoir(
            dim_reservoir=task_network.dim_reservoir,
            dim_input=task_network.dim_input + task_network.dim_output,
            dim_output=task_network.dim_output,
            tau=task_network.tau,
            chaos_factor=1.0,
            probability_recurrent_connection=1.0,
            device=self.device
        )

        # Some initialisations options from the FullFORCE paper
        if clone_input_weights:
            self.target_network.W_in[:, :self.task_network.dim_input] = self.task_network.W_in[:,
                                                                        :self.task_network.dim_input]

        if clone_target_weights:
            self.target_network.W_in[:, self.task_network.dim_input:] = self.task_network.W_in[:,
                                                                        :self.task_network.dim_output]

        if set_recurrent_weights_to_zeros:
            self.task_network.W_rec.zero_()

        # For RLS update | target network target weights remain fixed
        self.W_in_target = self.target_network.W_in[:, self.task_network.dim_input:]

    @torch.no_grad()
    def train_step(self,
                   input_signal: torch.Tensor,
                   target: torch.Tensor,
                   dt: float = 0.1,
                   sparce_update: bool = False):
        input_signal = input_signal.to(self.device)
        target = target.to(self.device)

        # Run networks forward
        target_network_s_k = self.target_network.forward(torch.cat([input_signal, target]), dt, compute_output=False)
        output, s_k = self.task_network.forward(input_signal, dt)

        # Compute errors
        J_err = (torch.matmul(self.task_network.W_rec, s_k) -
                 torch.matmul(self.target_network.W_rec, target_network_s_k) -
                 torch.matmul(self.W_in_target, target))

        error_y = output - target

        # Update P matrix
        Ps = torch.matmul(self.P, s_k)
        sPs = torch.matmul(s_k, Ps)
        k = Ps / (1 + sPs)
        self.P -= torch.outer(Ps, k)

        # Update weights and thresholds
        self.task_network.W_rec -= torch.outer(J_err, k)
        self.task_network.W_out -= torch.outer(error_y, k)

        if sparce_update:
            # Theta update after SpaRCe Paper
            # Term 1: Correlation between weighted activities
            delta_theta_1 = torch.zeros(self.task_network.dim_reservoir).to(self.device)
            for j in range(self.task_network.dim_output):
                weighted_activities = torch.matmul(self.task_network.W_out[j], s_k)
                delta_theta_1 += weighted_activities * self.task_network.W_out[j]

            # Term 2: Direct contribution to correct classification
            delta_theta_2 = -torch.matmul(error_y, self.task_network.W_out) * torch.sign(s_k)

            # Combine updates with RLS gain and update
            self.task_network.theta_tilde -= (delta_theta_1 + delta_theta_2) * k
        else:
            # Don't know if this is the correct way to update
            self.task_network.theta_tilde -= torch.sum(error_y, dim=0) * k

        return error_y

    def reset_states(self):
        self.task_network.reset_state()
        self.target_network.reset_state()
