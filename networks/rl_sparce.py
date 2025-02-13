import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .sparse_esn import SpaRCeESN
from .utils import z_normalize


# TODO: Add support for batch training
class RLSpaRCe(SpaRCeESN):
    def __init__(self,
                 dim_reservoir: int,
                 dim_input: int,
                 dim_output: int,
                 gamma: float = 0.99,  # discount factor
                 lambda_trace: float = 0.9,  # eligibility trace decay
                 **kwargs):
        super().__init__(dim_reservoir=dim_reservoir,
                         dim_input=dim_input,
                         dim_output=dim_output,
                         mode='regression',
                         **kwargs)

        self.gamma = gamma
        self.lambda_trace = lambda_trace

        # Eligibility traces
        self.e_w = None  # Will be initialized with batch size
        self.e_theta = None
        self.e_v = None

        # Value head for TD learning
        self.W_v = nn.Parameter(torch.zeros(1, dim_reservoir))

    def initialize_traces(self, batch_size: int):
        """Initialize eligibility traces with proper dimensions"""
        self.e_w = torch.zeros(batch_size, self.dim_output, self.dim_reservoir).to(self.device)
        self.e_theta = torch.zeros(batch_size, self.dim_reservoir).to(self.device)
        self.e_v = torch.zeros(batch_size, 1, self.dim_reservoir).to(self.device)

    def reset_traces(self, batch_indices: Optional[torch.Tensor] = None):
        """Reset eligibility traces for specified batch indices or all if None"""
        if batch_indices is None:
            self.e_w.zero_()
            self.e_theta.zero_()
            self.e_v.zero_()
        else:
            self.e_w[batch_indices] = 0
            self.e_theta[batch_indices] = 0
            self.e_v[batch_indices] = 0

    def compute_initial_percentiles(self, envs: List, num_steps: int = 1000):
        """
        Compute initial percentile thresholds by running random actions
        through the reservoir and collecting activity statistics
        """
        print("Computing initial activity percentiles...")
        batch_size = len(envs)
        V_history = []

        # Reset all environments and model
        self.reset_state(batch_size)

        # Collect reservoir activity statistics
        for _ in range(num_steps):
            # random actions
            actions = torch.randn(batch_size, self.dim_output).to(self.device)
            actions = torch.tanh(actions)

            # Step environments
            next_states = []
            for i, env in enumerate(envs):
                next_state, _, _ = env.step(actions[i].cpu().numpy())
                next_states.append(next_state)
            states = torch.from_numpy(np.array(next_states)).float().to(self.device)

            # Get reservoir activity
            V = self.forward_V(states)
            V_history.append(V)

        # Compute percentiles from collected activity
        V_history = torch.cat(V_history, dim=0)
        self.compute_percentile_thresholds(V_history)
        print("Percentile thresholds initialized.")

    def get_value(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute value estimate from current state"""
        if x is None:
            x = self.x
        return torch.matmul(x, self.W_v.t())

    def update_traces(self, actions: torch.Tensor):
        """
        Update eligibility traces using current activity and action

        The eligibility traces implement a form of credit assignment:
        - e_w tracks which synaptic connections contributed to the action
        - e_theta tracks which neurons were active and by how much
        - e_v tracks which neurons contributed to the value estimate
        """
        batch_size = actions.size(0)

        # Initialize traces if not done yet
        if self.e_w is None:
            self.initialize_traces(batch_size)

        # Update weight eligibility [batch_size, output_dim, reservoir_dim]
        self.e_w = self.lambda_trace * self.e_w + \
                   torch.bmm(actions.unsqueeze(2), self.x.unsqueeze(1))

        # Update threshold eligibility [batch_size, reservoir_dim]
        V_abs = torch.abs(self.V)
        self.e_theta = self.lambda_trace * self.e_theta + \
                       (V_abs - self.theta_tilde) * torch.sign(self.V)

        # Update value eligibility [batch_size, 1, reservoir_dim]
        self.e_v = self.lambda_trace * self.e_v + self.x.unsqueeze(1)

    def td_error(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute TD error for current transition
        :param rewards: reward for current transitions
        :param dones: True if episode is done, False otherwise
        """
        current_value = self.get_value().squeeze()

        # Zero value for terminal states
        next_value = torch.where(
            dones,
            torch.zeros_like(current_value),
            self.get_value().detach().squeeze()
        )

        return rewards + self.gamma * next_value - current_value

    def update_weights(self, td_errors: torch.Tensor, gradient_clip: bool = True):
        # Normalize TD error for stability
        td_errors_scaled = z_normalize(td_errors)
        # From [batch_size] to [batch_size, 1, 1] to align with eligibility traces
        td_errors_scaled = td_errors_scaled.view(-1, 1, 1)

        # Update readout weights
        with torch.no_grad():
            weight_updates = (td_errors_scaled * self.e_w).mean(dim=0)
            if gradient_clip:
                weight_updates = torch.clamp(weight_updates, -1, 1)
            self.W_o += self.lr_readout * weight_updates

        # Update thresholds
        with torch.no_grad():
            theta_updates = (td_errors_scaled.squeeze(-1) * self.e_theta).mean(dim=0)
            if gradient_clip:
                theta_updates = torch.clamp(theta_updates, -1, 1)
            self.theta_tilde += self.lr_threshold * theta_updates

        # Update value weights
        with torch.no_grad():
            value_updates = (td_errors_scaled * self.e_v).mean(dim=0)
            if gradient_clip:
                value_updates = torch.clamp(value_updates, -1, 1)
            self.W_v += self.lr_readout * value_updates

    def select_actions(self, epsilon: float = 0.1, std: float = 1.0) -> torch.Tensor:
        """
        Select action using epsilon-greedy exploration.
        TODO: Maybe change dynamics in the reservoir for exploration?
        """
        # Get network actions
        network_actions = torch.tanh(self.output)

        if epsilon > 0.0:
            batch_size = self.output.size(0)
            random_mask = torch.rand(batch_size).to(self.device) < epsilon

            # generate random actions
            random_actions = torch.tanh(std * torch.randn_like(network_actions))

            # Combine using mask
            actions = torch.where(
                random_mask.unsqueeze(1).expand_as(network_actions),
                random_actions,
                network_actions
            )
            return actions
        else:
            return network_actions

    def train_step(self, states: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """
        Run single training step and return selected actions

        :param states: tensor of shape [batch_size, input_dim]
        :param epsilon: exploration rate
        :return: actions of shape [batch_size, output_dim]
        """
        # Forward pass through network
        self.forward(states)

        # Select actions
        actions = self.select_actions(epsilon=epsilon)

        # Update eligibility traces
        self.update_traces(actions)

        return actions

    def test_step(self, states: torch.Tensor) -> torch.Tensor:
        # Forward pass through network
        self.forward(states)

        # Select actions
        return self.select_actions(epsilon=0.0)

    def update(self,
               rewards: torch.Tensor,
               next_states: torch.Tensor,
               dones: torch.Tensor):
        """
        Update network using reward signals

        :param rewards: tensor of shape [batch_size]
        :param next_states: tensor of shape [batch_size, input_dim]
        :param dones: tensor of shape [batch_size]
        """

        # Compute TD errors
        td_errors = self.td_error(rewards, dones)

        # Update weights using TD errors
        self.update_weights(td_errors)

        # Reset traces for done episodes
        if dones.any():
            self.reset_traces(dones)

        # Forward pass for next states
        if not dones.all():
            self.forward(next_states)
