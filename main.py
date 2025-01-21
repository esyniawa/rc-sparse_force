import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Type, Optional


def generate_periodic_signal(n_samples: int,
                             n_components: int,
                             dim: int,
                             base_period: float,
                             amplitude_range: Tuple[float, float] = (0.1, 1.0),
                             freq_range: Tuple[float, float] = (1.0, 2.0),
                             seed: Optional[int] = None) -> torch.Tensor:
    """Generate a periodic signal with superimposed sine and cosine waves."""
    t = np.linspace(0, base_period, n_samples)
    signal = np.zeros((n_samples, dim))

    for d in range(dim):
        current_signal = np.zeros(n_samples)
        dim_seed = None if seed is None else seed + d

        # Add sine components
        for i in range(n_components):
            if dim_seed is not None:
                np.random.seed(dim_seed + i)

            amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
            freq = np.random.uniform(freq_range[0], freq_range[1])
            current_signal += amplitude * np.sin(2 * np.pi * freq * t / base_period)

        signal[:, d] = current_signal / np.max(np.abs(current_signal))

    return torch.from_numpy(signal).float()


def plot_periodic_signal(signal: torch.Tensor, dt: float = 0.1, title: str = "Periodic Signal"):
    """Plot a periodic signal with multiple dimensions."""
    n_samples, dim = signal.shape
    t = np.arange(n_samples) * dt

    fig, axes = plt.subplots(dim, 1, figsize=(15, 3 * dim))
    if dim == 1:
        axes = [axes]

    for d in range(dim):
        axes[d].plot(t, signal[:, d], label=f'Dimension {d}')
        axes[d].set_xlabel('Time')
        axes[d].set_ylabel('Amplitude')
        axes[d].legend()
        axes[d].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def train_reservoir(trainer,
                    input_signal: torch.Tensor,
                    target_signal: torch.Tensor,
                    dt: float = 0.1,
                    print_interval: int = 100) -> List[float]:
    """
    Train a reservoir on given input and target signals.
    """

    n_steps = len(input_signal)
    mse_history = []

    print("Starting training...")
    for step in range(n_steps):
        error_minus = trainer.train_step(
            input_signal[step],
            target_signal[step],
            dt=dt
        )
        mse = torch.mean(error_minus ** 2).item()
        mse_history.append(mse)

        if (step + 1) % print_interval == 0:
            print(f"Step {step + 1}/{n_steps}, MSE: {mse:.6f}")

    return mse_history


def test_reservoir(reservoir,
                   input_signal: torch.Tensor,
                   target_signal: torch.Tensor,
                   dt: float = 0.1) -> Tuple[float, np.ndarray]:
    """
    Test a trained reservoir on given input and target signals.
    """
    print("\nStarting testing...")
    test_outputs = []
    reservoir.reset_state()

    for step in range(len(input_signal)):
        output, _ = reservoir.forward(input_signal[step], dt=dt)
        test_outputs.append(output.cpu().detach().numpy())

    test_outputs = np.array(test_outputs)
    target_np = target_signal.cpu().numpy()

    test_mse = np.mean((test_outputs - target_np) ** 2)

    return test_mse, test_outputs


def visualize_results(target_signal: np.ndarray,
                      output_signal: np.ndarray,
                      dt: float = 0.1,
                      mse_history: Optional[List[float]] = None):
    """
    Visualize the test results and optionally the training history.
    """

    # Plot signal comparison
    n_steps, output_dim = target_signal.shape
    t = np.arange(n_steps) * dt

    fig, axes = plt.subplots(output_dim, 1, figsize=(15, 3 * output_dim))
    if output_dim == 1:
        axes = [axes]

    for dim in range(output_dim):
        axes[dim].plot(t, target_signal[:, dim], 'b-', label='Target', alpha=0.7)
        axes[dim].plot(t, output_signal[:, dim], 'r--', label='Output', alpha=0.7)
        axes[dim].set_xlabel('Time')
        axes[dim].set_ylabel(f'Dimension {dim}')
        axes[dim].legend()
        axes[dim].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot training history if provided
    if mse_history is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(mse_history)
        plt.xlabel('Training Step')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)
        plt.title('Training MSE History')
        plt.show()


def run_experiment(reservoir_class: Type,
                   trainer_class: Type,
                   reservoir_params: dict,
                   trainer_params: dict,
                   input_dim: int,
                   output_dim: int,
                   n_input_components: int,
                   n_output_components: int,
                   training_steps: int,
                   testing_steps: int,
                   base_period: float = 100.0,
                   dt: float = 0.1,
                   seed: Optional[int] = None,
                   device: Optional[torch.device] = None):
    """
    Run a complete reservoir computing experiment.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate signals
    input_signal = generate_periodic_signal(
        n_samples=max(training_steps, testing_steps),
        n_components=n_input_components,
        dim=input_dim,
        base_period=base_period,
        seed=seed
    ).to(device)

    target_signal = generate_periodic_signal(
        n_samples=max(training_steps, testing_steps),
        n_components=n_output_components,
        dim=output_dim,
        base_period=base_period,
        seed=None if seed is None else seed + input_dim * n_input_components
    ).to(device)

    # Plot example signals
    plot_periodic_signal(input_signal, dt=dt, title="Input Signal")
    plot_periodic_signal(target_signal, dt=dt, title="Target Signal")

    # Initialize models
    reservoir = reservoir_class(**reservoir_params).to(device)
    trainer = trainer_class(reservoir, **trainer_params)

    # Train
    mse_history = train_reservoir(
        trainer=trainer,
        input_signal=input_signal[:training_steps],
        target_signal=target_signal[:training_steps],
        dt=dt
    )

    # Test
    test_mse, test_outputs = test_reservoir(
        reservoir=reservoir,
        input_signal=input_signal[:testing_steps],
        target_signal=target_signal[:testing_steps],
        dt=dt
    )

    # Visualize results
    visualize_results(
        target_signal=target_signal[:testing_steps].cpu().numpy(),
        output_signal=test_outputs,
        dt=dt,
        mse_history=mse_history
    )

    print(f"\nFinal test MSE: {test_mse:.6f}")

    return test_mse, mse_history, test_outputs


# Example usage:
if __name__ == "__main__":
    from network.reservoir_torch import (SparseReservoir,
                                         SparseFullForceTrainer,
                                         Reservoir,
                                         FullForceTrainer,
                                         ForceTrainer)

    # Example parameters
    reservoir_params = {
        "dim_reservoir": 500,
        "dim_input": 2,
        "dim_output": 2,
        "tau": 0.01,
        "probability_recurrent_connection": 1.0

    }

    trainer_params = {
        "alpha": 1.0
    }

    test_mse, mse_history, outputs = run_experiment(
        reservoir_class=Reservoir,
        trainer_class=FullForceTrainer,
        reservoir_params=reservoir_params,
        trainer_params=trainer_params,
        input_dim=2,
        output_dim=2,
        n_input_components=3,
        n_output_components=3,
        training_steps=4_000,
        testing_steps=1000,
        base_period=100.0,
        dt=0.001,
        seed=42
    )
