import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from typing import Tuple, Optional
from networks.sparse_esn import SpaRCeESN, SpaRCeLoss


class SequentialMNIST(Dataset):
    """Dataset wrapper that returns MNIST images column by column"""

    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        # Load MNIST dataset
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=True
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image and label
        image, label = self.mnist[idx]

        # Convert image to sequential format (28 time steps of 28-dimensional vectors)
        # Image shape goes from (1, 28, 28) to (28, 28)
        seq_image = image.squeeze(0).t()  # Transpose to get columns

        # Apply additional transforms if specified
        if self.transform is not None:
            seq_image = self.transform(seq_image)

        return seq_image, label


def train_evaluate_sparce_mnist(
        model: SpaRCeESN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        device: torch.device,
        weight_decay_readout: float = 1e-5,
        subset_size: Optional[int] = 64,  # How many batches to use for percentile computation (can eat up memory pretty quickly)
        print_interval: int = 250
) -> Tuple[list, list]:
    """Train and evaluate SpaRCe model on sequential MNIST"""
    criterion = SpaRCeLoss(loss_type='sigmoidal_cross_entropy')

    # Optimizer for readout weights
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr_readout, weight_decay=weight_decay_readout)

    # Add learning rate scheduler for readout weights
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor accuracy
        factor=0.5,  # reduce lr by half
        patience=2,  # Wait 2 epochs
        min_lr=1e-7
    )

    # Lists to store training and test accuracies
    train_accuracies = []
    test_accuracies = []

    # Compute initial activity percentiles
    print("Computing initial activity percentiles...")
    V_batch = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            if subset_size is not None and batch_idx > subset_size:
                break
            data = data.to(device)
            batch_size = data.size(0)
            n_cols = data.size(1)
            model.reset_state(batch_size)

            for t in range(n_cols):  # Process each column
                V = model.forward_V(data[:, t, :])
                V_batch.append(V)

    V_batch = torch.cat(V_batch, dim=0)
    model.compute_percentile_thresholds(V_batch)
    # rm V_batch
    del V_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Convert targets to one-hot
            target_onehot = torch.zeros(batch_size, model.dim_output, device=device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)

            # Reset reservoir state
            model.reset_state(batch_size)

            # Process each column of the images
            for t in range(n_cols):  # 28 time steps
                output = model(data[:, t, :])

            # Compute loss on final output
            loss = criterion(output, target_onehot)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.W_o, max_norm=1.0)  # Clip gradients
            optimizer.step()

            # Update thresholds
            model.update_thresholds(target_onehot)

            # Compute accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += batch_size

            if (batch_idx + 1) % print_interval == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}, '
                      f'Accuracy: {100.0 * correct / total:.2f}%')

        # Compute epoch training accuracy
        train_acc = 100.0 * correct / total
        train_accuracies.append(train_acc)

        # Evaluate on test set
        test_acc = evaluate_sparce_mnist(model, test_loader, device)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch}: Train Accuracy = {train_acc:.2f}%, Test Accuracy = {test_acc:.2f}%')

        # Step scheduler and update readout learning rates
        prev_lr = scheduler.get_last_lr()[0]
        scheduler.step(test_acc)
        new_lr = scheduler.get_last_lr()[0]

        # If update by scheduler, update threshold learning rate proportionally
        if new_lr != prev_lr:
            lr_scale = new_lr / prev_lr
            model.lr_threshold = model.lr_threshold * lr_scale

    return train_accuracies, test_accuracies


def evaluate_sparce_mnist(
        model: SpaRCeESN,
        test_loader: DataLoader,
        device: torch.device
) -> float:
    """Evaluate SpaRCe model on MNIST test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            n_cols = data.size(1)

            # Reset reservoir state
            model.reset_state(batch_size)

            # Process each column
            for t in range(n_cols):
                output = model(data[:, t, :])

            # Compute accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += batch_size

    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == "__main__":
    from plotting_functions import plot_errors
    from arguments_for_runs import get_training_args, save_args

    args, _ = get_training_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # Create datasets
    train_dataset = SequentialMNIST(root='./mnist_data', train=True)
    test_dataset = SequentialMNIST(root='./mnist_data', train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SpaRCeESN(
        dim_reservoir=args.dim_res,
        dim_input=28,  # Each MNIST column is 28 pixels
        dim_output=10,  # 10 digits
        mode='classification',
        percentile_n=args.perc_n,
        probability_recurrent_connection=args.prop_rec,
        spectral_radius=args.spectral_radius,
        learning_rate_threshold=5e-5,
        learning_rate_readout=1e-4,
        feedforward_scaling=args.ff_scale,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
    )

    # Train and evaluate
    train_accs, test_accs = train_evaluate_sparce_mnist(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=args.device
    )

    print(f"Final test accuracy: {test_accs[-1]:.2f}%")
    results_folder = f'results/MNIST_simulation_{args.sim_id}'
    os.makedirs(results_folder, exist_ok=True)

    # Plot training and test accuracy
    plot_errors(train_accs, test_accs,
                save_name=f'{results_folder}/accuracy.png')

    # Save arguments
    save_args(args, save_name=f'{results_folder}/simulation_params.txt')

    # Save model
    model.save_model(f'{results_folder}/model.pth')
