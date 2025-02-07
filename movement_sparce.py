import torch
from typing import Optional
from networks.sparse_esn import SpaRCeESN, SpaRCeLoss, mse_loss
from generate_movement_dataset import PlanarArmDataset, PlanarArmDataLoader
from plotting_functions import plot_errors, plot_batch_predictions


def train_evaluate_sparce_arm(
        model: SpaRCeESN,
        train_loader: PlanarArmDataLoader,
        test_loader: PlanarArmDataLoader,
        num_epochs: int,
        device: torch.device,
        weight_decay_readout: float = 1e-5,
        subset_size: int = 64,
        print_interval: Optional[int] = None,
        plot_folder: Optional[str] = None
) -> tuple[list, list]:
    """Train and evaluate SpaRCe model on planar arm control."""

    criterion = SpaRCeLoss(loss_type='mse')  # Use MSE loss for regression

    # Optimizer for readout weights
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr_readout, weight_decay=weight_decay_readout)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Monitor MSE loss
        factor=0.5,
        patience=2,
        min_lr=1e-8
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
            n_steps = data.size(1)
            model.reset_state(batch_size)

            # Process full trajectory
            for t in range(n_steps):
                V = model.forward_V(data[:, t, :])
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
            n_steps = data.size(1)
            batch_mse = 0

            # Reset reservoir state
            model.reset_state(batch_size)
            for t in range(n_steps):
                # Forward pass
                output = model(data[:, t, :])

                # Compute MSE loss
                loss = criterion(output, target[:, t, :])
                batch_mse += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.W_o, max_norm=1.0)
                optimizer.step()

                # Update thresholds
                model.update_thresholds(target[:, t, :])

            # Track MSE
            total_mse += batch_mse
            num_batches += 1

            if print_interval is not None and batch_idx % print_interval == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'MSE = {batch_mse:.6f}')


        # Compute epoch training MSE
        train_mse = total_mse / num_batches
        train_mses.append(train_mse)

        # Evaluate on test set
        if plot_folder is not None:
            epoch_folder = plot_folder + f'/epoch_{epoch}/'
        else:
            epoch_folder = None
        test_mse = evaluate_sparce_arm(model, test_loader, device, plot_folder=epoch_folder)
        test_mses.append(test_mse)

        print(f'Epoch {epoch}: Train MSE = {train_mse:.6f}, '
              f'Test MSE = {test_mse:.6f}')

        # Step scheduler
        scheduler.step(test_mse)

    return train_mses, test_mses


def evaluate_sparce_arm(
        model: SpaRCeESN,
        test_loader: PlanarArmDataLoader,
        device: torch.device,
        plot_folder: Optional[str] = None,
) -> float:
    """Evaluate SpaRCe model on arm control test set."""
    model.eval()
    total_mse = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            n_steps = data.size(1)
            batch_mse = 0

            if plot_folder is not None:
                # Initialize arrays to store outputs
                batch_outputs = torch.zeros((batch_size, n_steps, 2), device=device)

            # Reset reservoir state
            model.reset_state(batch_size)

            for t in range(n_steps):
                # Forward pass
                output = model(data[:, t, :])

                if plot_folder is not None:
                    batch_outputs[:, t, :] = output

                # Compute MSE
                mse = mse_loss(output, target[:, t, :]).item()
                batch_mse += mse

            if plot_folder is not None:
                plot_batch_predictions(
                    batch_outputs=batch_outputs,
                    batch_targets=target,
                    dataset=test_loader.dataset,
                    batch_idx=num_batches,
                    save_name=plot_folder + f'/predictions_batch_{num_batches}.pdf',
                    inverse_transform=False,
                )

            total_mse += batch_mse
            num_batches += 1

    return total_mse / num_batches


if __name__ == "__main__":
    from arguments_for_runs import get_training_args, save_args, import_args

    # Get arguments
    args, _ = get_training_args()
    data_args = import_args(args.data_set_dir + '/dataset_args.txt')
    result_folder = f'results/arm_control_{args.sim_id}/'

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create dataset
    dataset = PlanarArmDataset(
        num_t=int(data_args.num_t),
        num_init_thetas=int(data_args.num_init_thetas),
        num_goals=int(data_args.num_goals),
        wait_steps_after_trajectory=int(data_args.wait_steps),
        movement_duration=float(data_args.movement_duration),
        train_split=float(data_args.train_split),
        save_dir=args.data_set_dir,
    )

    # Create data loaders
    train_loader = PlanarArmDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        mode='train',
        num_episodes=args.num_train_episodes,
        shuffle=True,
        shuffle_episodes=True,
    )
    test_loader = PlanarArmDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        mode='eval',
        num_episodes=args.num_eval_episodes,
        shuffle=True,
        shuffle_episodes=True,
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
        device=args.device,
        plot_folder=result_folder,
    )

    # Make final test over the entire test set
    test_loader.set_num_episodes(None)
    final_mse = evaluate_sparce_arm(model=model,
                                    test_loader=test_loader,
                                    device=args.device,
                                    plot_folder=None)

    print(f"Final test MSE: {final_mse:.6f}")
    # Save results
    save_args(args,
              save_name=result_folder + 'parameters.txt',
              additional_args={'final_mse_test': final_mse})

    # Plot training + eval results
    plot_errors(train_mses, test_mses, save_name=result_folder + 'errors.pdf')
