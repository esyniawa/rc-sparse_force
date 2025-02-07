import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from generate_movement_dataset import PlanarArmDataset
from typing import Optional, Tuple, List


def save_or_show(save_name: Optional[str], fig: plt.Figure):
    if save_name is not None:
        save_folder = os.path.split(save_name)[0]
        if save_folder != '': os.makedirs(save_folder, exist_ok=True)
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    else:
        fig.show()

    plt.close()


def plot_errors(train_errors: np.ndarray | List[float],
                test_errors: np.ndarray | List[float],
                save_name: Optional[str],
                fig_size: Tuple[int, int] = (10, 6)):
    fig = plt.figure(figsize=fig_size)
    plt.plot(train_errors, label='Train', color='green', linestyle='--')
    plt.plot(test_errors, label='Test', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    save_or_show(save_name, fig)


def plot_batch_predictions(batch_outputs: torch.Tensor,
                           batch_targets: torch.Tensor,
                           dataset: PlanarArmDataset,
                           inverse_transform: bool,
                           batch_idx: int,
                           max_plot_size: int = 32,
                           save_name: Optional[str] = None):
    """
    Plot the predictions and targets for each sequence in a batch.

    :param batch_outputs: Model outputs (batch_size, n_steps, 2)
    :param batch_targets: Target values (batch_size, n_steps, 2)
    :param dataset: Dataset object for inverse transformation
    :param inverse_transform: Rescale predictions and targets
    :param batch_idx: Index of the current batch
    :param max_plot_size: Maximum number of sequences to plot
    :param save_name: Path and name of the file. If None, show the plot
    """
    batch_size = batch_outputs.shape[0]

    # select a subset of the batch if it is too large
    if batch_size > max_plot_size:
        indices = np.random.choice(batch_size, max_plot_size, replace=False)
        batch_outputs = batch_outputs[indices]
        batch_targets = batch_targets[indices]
        plot_size = max_plot_size
    else:
        plot_size = batch_size

    if inverse_transform:
        # Convert to numpy and inverse transform
        outputs_np = dataset.inverse_transform_targets(batch_outputs)
        targets_np = dataset.inverse_transform_targets(batch_targets)
    else:
        # Convert to numpy
        outputs_np = batch_outputs.detach().cpu().numpy()
        targets_np = batch_targets.detach().cpu().numpy()

    # Calculate number of rows and columns for subplots
    n_cols = min(8, plot_size)
    n_rows = (plot_size + n_cols - 1) // n_cols

    # Create figure
    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.suptitle(f'Batch {batch_idx + 1} - Predictions vs Targets', fontsize=16)

    # Create subplots for each sequence in the batch
    for i in range(batch_size):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        ax.plot(outputs_np[i, :, 0], label='Pred Joint 1', color='blue', linestyle='--')
        ax.plot(outputs_np[i, :, 1], label='Pred Joint 2', color='red', linestyle='--')

        ax.plot(targets_np[i, :, 0], label='Target Joint 1', color='blue')
        ax.plot(targets_np[i, :, 1], label='Target Joint 2', color='red')

        ax.set_title(f'Sequence {i + 1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Velocity')
        ax.grid(True)
        if i == 0:  # Only show legend for first subplot
            ax.legend()

    save_or_show(save_name, fig)
