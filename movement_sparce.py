import argparse
import torch
import numpy as np
from networks.sparse_esn import SpaRCeESN, SpaRCeLoss
from generate_movement_dataset import PlanarArmDataset, PlanarArmDataLoader


def get_training_args():
    """Define and parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description='Train SpaRCe for Planar Arm Control')

    # Model parameters
    parser.add_argument('--dim_res', type=int, default=1000,
                        help='Reservoir dimension')
    parser.add_argument('--perc_n', type=float, default=75.0,
                        help='Percentile threshold')
    parser.add_argument('--prop_rec', type=float, default=0.01,
                        help='Proportion of recurrent connections')
    parser.add_argument('--spectral_radius', type=float, default=0.97,
                        help='Spectral radius')
    parser.add_argument('--ff_scale', type=float, default=0.1,
                        help='Scale of feedforward connections')
    parser.add_argument('--alpha', type=float, default=0.17,
                        help='Alpha parameter for leaky integrator')

    # Dataset parameters
    parser.add_argument('--num_t', type=int, default=110,
                        help='Number of time steps')
    parser.add_argument('--num_init_thetas', type=int, default=500,
                        help='Number of initial joint angles')
    parser.add_argument('--num_goals', type=int, default=500,
                        help='Number of target positions')
    parser.add_argument('--wait_steps', type=int, default=10,
                        help='Wait steps after trajectory')
    parser.add_argument('--movement_duration', type=float, default=5.0,
                        help='Duration of movement')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/eval split')
    parser.add_argument('--save_dir', type=str, default='arm_data',
                        help='Directory to save dataset')

    # Training parameters
    parser.add_argument('--sim_id', type=int, default=0,
                        help='Simulation id')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


def train_evaluate_sparce_arm(
        model: SpaRCeESN,
        train_loader: PlanarArmDataLoader,
        test_loader: PlanarArmDataLoader,
        num_epochs: int,
        device: torch.device,
        weight_decay_readout: float = 1e-5,
        subset_size: int = 512,
        print_interval: int = 250
) -> tuple[list, list]:
    """Train and evaluate SpaRCe model on planar arm control."""

    criterion = SpaRCeLoss(loss_type='mse')  # Use MSE loss for regression

    # Optimizer for readout weights
    optimizer = torch.optim.Adam([
        {
            'params': model.W_o,
            'weight_decay': weight_decay_readout
        }
    ], lr=model.lr_readout)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor MSE loss
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # Lists to store training and test MSE
    train_mses = []
    test_mses = []

    # Compute initial activity percentiles
    print("Computing initial activity percentiles...")
    V_batch = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            if subset_size is not None and batch_idx > subset_size:
                break
            data = data.to(device)
            batch_size = data.size(0)
            model.reset_state(batch_size)

            # Process full trajectory
            V = model.forward_V(data)
            V_batch.append(V)

    V_batch = torch.cat(V_batch, dim=0)
    model.compute_percentile_thresholds(V_batch)
    del V_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_mse = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Reset reservoir state
            model.reset_state(batch_size)

            # Forward pass
            output = model(data)

            # Compute MSE loss
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.W_o, max_norm=1.0)
            optimizer.step()

            # Update thresholds
            model.update_thresholds(target)

            # Track MSE
            total_mse += loss.item()
            num_batches += 1

            if (batch_idx + 1) % print_interval == 0:
                current_mse = total_mse / num_batches
                print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, '
                      f'MSE: {current_mse:.6f}')

        # Compute epoch training MSE
        train_mse = total_mse / num_batches
        train_mses.append(train_mse)

        # Evaluate on test set
        test_mse = evaluate_sparce_arm(model, test_loader, device)
        test_mses.append(test_mse)

        print(f'Epoch {epoch}: Train MSE = {train_mse:.6f}, '
              f'Test MSE = {test_mse:.6f}')

        # Step scheduler
        scheduler.step(test_mse)

    return train_mses, test_mses


def evaluate_sparce_arm(
        model: SpaRCeESN,
        test_loader: PlanarArmDataLoader,
        device: torch.device
) -> float:
    """Evaluate SpaRCe model on arm control test set."""
    model.eval()
    total_mse = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Reset reservoir state
            model.reset_state(batch_size)

            # Forward pass
            output = model(data)

            # Compute MSE
            mse = ((target - output) ** 2).mean().item()
            total_mse += mse
            num_batches += 1

    return total_mse / num_batches


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Get arguments
    args = get_training_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create dataset
    dataset = PlanarArmDataset(
        num_t=args.num_t,
        num_init_thetas=args.num_init_thetas,
        num_goals=args.num_goals,
        wait_steps_after_trajectory=args.wait_steps,
        movement_duration=args.movement_duration,
        train_split=args.train_split,
        save_dir=args.save_dir
    )

    # Create data loaders
    train_loader = PlanarArmDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        mode='train'
    )
    test_loader = PlanarArmDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        mode='eval'
    )

    # Initialize model
    model = SpaRCeESN(
        dim_reservoir=args.dim_res,
        dim_input=4,  # Current joint angles (2) + goal position (2)
        dim_output=2,  # Delta joint angles (2)
        mode='regression',
        percentile_n=args.perc_n,
        probability_recurrent_connection=args.prop_rec,
        spectral_radius=args.spectral_radius,
        learning_rate_threshold=5e-5,
        learning_rate_readout=1e-4,
        feedforward_scaling=args.ff_scale,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device
    )

    # Train and evaluate
    train_mses, test_mses = train_evaluate_sparce_arm(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        device=args.device
    )

    print(f"Final test MSE: {test_mses[-1]:.6f}")
    results_folder = f'results/arm_simulation_{args.sim_id}'
    os.makedirs(results_folder, exist_ok=True)

    # Plot training and test accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_mses, label='Train', color='green', linestyle='--')
    plt.plot(test_mses, label='Test', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'{results_folder}/movement_sparce_dim[{args.dim_res}]_percentile[{args.perc_n}]_prop_rec[{args.prop_rec}].pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Save parameters
    with open(f'{results_folder}/movement_sparce_dim[{args.dim_res}]_percentile[{args.perc_n}]_prop_rec[{args.prop_rec}].txt', 'w') as f:
        f.write(f"Final test accuracy: {test_mses[-1]:.2f}%\n")
        for name, value in vars(args).items():
            f.write(f"{name}: {value}\n")
