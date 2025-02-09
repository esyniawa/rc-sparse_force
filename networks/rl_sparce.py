import torch
import torch.nn as nn
from typing import Optional, Tuple
from .sparse_esn import SpaRCeESN


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

        # Initialize eligibility traces
        self.e_w = torch.zeros_like(self.W_o)  # Weight eligibility
        self.e_theta = torch.zeros_like(self.theta_tilde)  # Threshold eligibility

        # Value head for TD learning
        self.W_v = nn.Parameter(torch.zeros(1, dim_reservoir))
        self.e_v = torch.zeros_like(self.W_v)  # Value eligibility

    def reset_traces(self):
        self.e_w.zero_()
        self.e_theta.zero_()
        self.e_v.zero_()

    def get_value(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = self.x
        return torch.matmul(x, self.W_v.t())

    def update_traces(self, action: torch.Tensor):
        # Update weight eligibility
        self.e_w = self.lambda_trace * self.e_w + torch.outer(action, self.x)

        # Update threshold eligibility
        V_abs = torch.abs(self.V)
        self.e_theta = self.lambda_trace * self.e_theta + \
                       (V_abs - self.theta_tilde) * torch.sign(self.V)

        # Update value eligibility
        self.e_v = self.lambda_trace * self.e_v + self.x

    def td_error(self, reward: float, done: bool) -> torch.Tensor:
        """
        Compute TD error for current transition
        :param reward: reward for current transition
        :param done: True if episode is done, False otherwise
        """
        current_value = self.get_value()

        # For terminal states, next value is 0
        if done:
            next_value = 0.0
        else:
            next_value = self.get_value().detach()

        return reward + self.gamma * next_value - current_value

    def update_weights(self, td_error: torch.Tensor):
        # Scale TD error for stability
        td_error_scaled = torch.tanh(td_error)  # Clip to [-1, 1]

        # Update readout weights
        with torch.no_grad():
            self.W_o += self.lr_readout * td_error_scaled * self.e_w

        # Update thresholds
        with torch.no_grad():
            self.theta_tilde += self.lr_threshold * td_error_scaled * self.e_theta

        # Update value weights
        with torch.no_grad():
            self.W_v += self.lr_readout * td_error_scaled * self.e_v

    def select_action(self, epsilon: float = 0.1) -> torch.Tensor:
        """Select action using epsilon-greedy exploration. TODO: Maybe change dynamics in the reservoir for exploration?"""
        if torch.rand(1) < epsilon:
            # Random action using uniform distribution followed by tanh
            action = torch.tanh(torch.randn(self.dim_output))
        else:
            # Use tanh-bounded network output as action
            action = torch.tanh(self.output).squeeze()

        return action.to(self.device)

    def train_step(self, state: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """
        Run single training step and return selected action
        :param state: state of environment
        :param epsilon: exploration rate
        """
        # Forward pass through network
        self.forward(state)

        # Select action
        action = self.select_action(epsilon)

        # Update eligibility traces
        self.update_traces(action)

        return action

    def update(self, reward: float, next_state: torch.Tensor, done: bool):
        """
        Update network using reward signal
        """
        # Compute TD error
        td_error = self.td_error(reward, done)

        # Update weights using TD error
        self.update_weights(td_error)

        # If episode is done, reset eligibility traces
        if done:
            self.reset_traces()

        # Forward pass for next state
        if not done:
            self.forward(next_state)


"""
DEMO:
# Training loop
for episode in range(num_episodes):
    state = env.reset()
    agent.reset_state()
    done = False
    
    for step in range(max_steps):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Select action
        action = agent.train_step(state_tensor)
        
        # Take action in environment
        next_state, reward, done = env.step(action.cpu().numpy())
        
        # Update agent
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        agent.update(reward, next_state_tensor, done)
    
        # Update state
        state = next_state
        
        if done:
            break
"""