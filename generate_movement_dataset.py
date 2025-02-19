import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from typing import Tuple, Optional
from kinematics.planar_arm import PlanarArmTrajectory
from tqdm.auto import tqdm


class PlanarArmDataset(Dataset):
    def __init__(self,
                 arm: str = 'right',
                 num_t: int = 100,
                 num_init_thetas: int = 100,
                 num_goals: int = 50,
                 wait_steps_after_trajectory: int = 10,
                 movement_duration: float = 5.0,
                 train_split: float = 0.8,
                 save_dir: Optional[str] = None):
        """
        Initialize the dataset for the planar arm.

        :param num_t: Number of time steps of the trajectory
        :param num_init_thetas: Number of initial joint angles
        :param num_goals: Number of target positions
        :param wait_steps_after_trajectory: Number of time steps to wait after each trajectory
        :param movement_duration: Duration of the movement in seconds (for generating velocities)
        :param train_split: Fraction of the dataset to use for training
        :param save_dir: Directory to save / load the dataset
        """

        self.wait_steps = wait_steps_after_trajectory
        self.num_t = num_t + self.wait_steps
        self.num_init_thetas = num_init_thetas
        self.num_goals = num_goals
        self.movement_duration = movement_duration
        self.save_dir = save_dir

        # Initialize arm trajectory planner
        self.arm = PlanarArmTrajectory(arm=arm, num_ik_points=12, num_trajectory_points=num_t)

        # Initialize scalers
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

        # Generate or load dataset
        if save_dir and self._check_saved_data():
            self._load_dataset()
        else:
            self._generate_dataset()

        # Split dataset into train and evaluation sets
        self.train_split = train_split
        self.train_goals = int(self.num_goals * train_split)
        self.eval_goals = self.num_goals - self.train_goals

        # Create indices for sampling
        self._create_sampling_indices()

    def _check_saved_data(self) -> bool:
        """Check if saved dataset exists."""
        print(f"Loading dataset from folder '{self.save_dir}'...")
        if not self.save_dir:
            return False

        files = ['inputs.npy', 'targets.npy', 'input_scaler.pkl', 'target_scaler.pkl']
        return all(os.path.exists(os.path.join(self.save_dir, f)) for f in files)

    def _generate_random_init_thetas(self) -> np.ndarray:
        """Generate random initial joint angles within limits."""
        theta1 = np.random.uniform(
            self.arm.l_upper_arm_limit,
            self.arm.u_upper_arm_limit,
            size=self.num_init_thetas
        )
        theta2 = np.random.uniform(
            self.arm.l_forearm_limit,
            self.arm.u_forearm_limit,
            size=self.num_init_thetas
        )
        return np.column_stack((theta1, theta2))

    def _generate_random_goals(self) -> np.ndarray:
        """Generate random reachable target positions."""
        # Generate random angles within limits
        theta1 = np.random.uniform(
            self.arm.l_upper_arm_limit,
            self.arm.u_upper_arm_limit,
            size=self.num_goals
        )
        theta2 = np.random.uniform(
            self.arm.l_forearm_limit,
            self.arm.u_forearm_limit,
            size=self.num_goals
        )

        # Calculate end-effector positions for these angles
        goals = []
        for t1, t2 in zip(theta1, theta2):
            pos = self.arm.forward_kinematics(np.array([t1, t2]), radians=True)
            goals.append(pos[:, -1])

        return np.array(goals)

    def _generate_dataset(self):
        """Generate the full dataset."""
        # Generate initial configurations and goals
        init_thetas = self._generate_random_init_thetas()
        goals = self._generate_random_goals()

        # Initialize arrays for inputs and targets
        # Shape: (num_goals, num_init_thetas, num_t, 4/2)
        self.inputs = np.zeros((self.num_goals, self.num_init_thetas, self.num_t, 4))
        self.targets = np.zeros((self.num_goals, self.num_init_thetas, self.num_t, 2))

        # Generate trajectories for each combination
        tqdm.write("Generating trajectories...")
        for i, goal in tqdm(enumerate(goals), total=len(goals), desc="Goals"):
            for j, theta in tqdm(enumerate(init_thetas), total=len(init_thetas),
                                 desc=f"Trajectories for goal {i + 1}/{len(goals)}",
                                 leave=False):

                # Generate trajectory
                joint_traj, _ = self.arm.plan_trajectory(current_angles=theta, target_coord=goal, waiting_steps=self.wait_steps)

                # Calculate velocities (targets)
                velocities = self.arm.get_trajectory_velocities(joint_traj)

                # Store data
                # For each timestep, store current angles and goal
                self.inputs[i, j, :, :2] = joint_traj  # Current angles
                self.inputs[i, j, :, 2:] = goal  # Goal position (same for all timesteps)
                self.targets[i, j, :, :] = velocities  # Joint velocities

        # Reshape for scaling
        input_shape = self.inputs.shape
        target_shape = self.targets.shape

        # Fit scalers
        self.input_scaler.fit(self.inputs.reshape(-1, 4))
        self.target_scaler.fit(self.targets.reshape(-1, 2))

        # Transform data
        self.inputs = self.input_scaler.transform(
            self.inputs.reshape(-1, 4)
        ).reshape(input_shape)
        self.targets = self.target_scaler.transform(
            self.targets.reshape(-1, 2)
        ).reshape(target_shape)

        # Save dataset if directory is specified
        if self.save_dir:
            self._save_dataset()

    def _create_sampling_indices(self):
        """Create shuffled indices for each goal."""
        self.goal_indices = list(range(self.num_goals))
        self.theta_indices = {}
        for goal_idx in range(self.num_goals):
            self.theta_indices[goal_idx] = list(range(self.num_init_thetas))
            np.random.shuffle(self.theta_indices[goal_idx])

    def _save_dataset(self):
        """Save dataset and scalers to disk."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save numpy arrays
        np.save(os.path.join(self.save_dir, 'inputs.npy'), self.inputs)
        np.save(os.path.join(self.save_dir, 'targets.npy'), self.targets)

        # Save scalers
        with open(os.path.join(self.save_dir, 'input_scaler.pkl'), 'wb') as f:
            pickle.dump(self.input_scaler, f)
        with open(os.path.join(self.save_dir, 'target_scaler.pkl'), 'wb') as f:
            pickle.dump(self.target_scaler, f)

    def _load_dataset(self):
        """Load dataset and scalers from disk."""
        # Load numpy arrays
        self.inputs = np.load(os.path.join(self.save_dir, 'inputs.npy'))
        self.targets = np.load(os.path.join(self.save_dir, 'targets.npy'))

        # Load scalers
        with open(os.path.join(self.save_dir, 'input_scaler.pkl'), 'rb') as f:
            self.input_scaler = pickle.load(f)
        with open(os.path.join(self.save_dir, 'target_scaler.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)

    def __len__(self) -> int:
        """Return number of goals (each goal will be an epoch)."""
        return self.num_goals

    def __getitem__(self, goal_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all trajectories for one goal with randomly selected initial thetas.
        Returns complete time series for each trajectory.
        """
        # Get inputs and targets for the specific goal
        inputs = self.inputs[goal_idx]  # Shape: (num_init_thetas, num_t, 4)
        targets = self.targets[goal_idx]  # Shape: (num_init_thetas, num_t, 2)

        # Convert to torch tensors
        return (torch.FloatTensor(inputs),
                torch.FloatTensor(targets))

    def inverse_transform_targets(self, targets: np.ndarray | torch.Tensor) -> np.ndarray:
        """Transform scaled targets back to original scale."""
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # handle batches
        if targets.ndim >= 2:
            org_shape = targets.shape
            targets = targets.reshape(-1, targets.shape[-1])
            targets = self.target_scaler.inverse_transform(targets)
            return targets.reshape(org_shape)
        else:
            return self.target_scaler.inverse_transform(targets)


class PlanarArmDataLoader:
    train: str = 'train'
    eval: str = 'eval'
    def __init__(self,
                 dataset: PlanarArmDataset,
                 batch_size: int,
                 mode: str,
                 num_episodes: int = None,
                 shuffle: bool = True,
                 shuffle_episodes: bool = True):
        """
        Custom data loader for planar arm trajectories.

        :param dataset: PlanarArmDataset
        :param batch_size: Number of trajectories per batch (same goal, different init thetas)
        :param mode: Either 'train' or 'eval'
        :param num_episodes: Maximum number of episodes to use (if None, use all available)
        :param shuffle: Shuffle the order of the initial thetas within each episode
        :param shuffle_episodes: Shuffle the order of episodes between epochs
        """
        self.dataset = dataset
        self.batch_size = min(batch_size, dataset.num_init_thetas)
        self.shuffle = shuffle
        self.shuffle_episodes = shuffle_episodes
        self.goal_idx = 0

        # Set goals range based on mode
        if mode == self.train:
            self.start_goal = 0
            self.available_goals = dataset.train_goals
        elif mode == self.eval:
            self.start_goal = dataset.train_goals
            self.available_goals = dataset.eval_goals
        else:
            raise ValueError(f"Mode must be either '{self.train}' or '{self.eval}'")

        # Initialize episode indices
        self.set_num_episodes(num_episodes)

    def set_num_episodes(self, num_episodes: int = None) -> None:
        """
        Change the number of episodes used by the data loader.

        :param num_episodes: New number of episodes (if None, use all available)
        """
        # Update number of goals
        self.num_goals = min(num_episodes, self.available_goals) if num_episodes else self.available_goals

        # Reset and recreate episode indices
        self.episode_indices = list(range(self.start_goal, self.start_goal + self.num_goals))
        if self.shuffle_episodes:
            self._shuffle_episodes()

        # Reset iterator
        self.goal_idx = 0

    def _shuffle_episodes(self):
        """Shuffle the order of episodes"""
        np.random.shuffle(self.episode_indices)

    def __iter__(self):
        """Return iterator over batches."""
        if self.shuffle_episodes:
            self._shuffle_episodes()
        self.goal_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of trajectories."""
        if self.goal_idx >= self.num_goals:
            self.goal_idx = 0
            raise StopIteration

        # Get actual goal index from our shuffled indices
        actual_goal_idx = self.episode_indices[self.goal_idx]

        # Get all trajectories for current goal
        inputs, targets = self.dataset[actual_goal_idx]

        # Get random indices for batch
        if self.shuffle:
            indices = np.random.choice(
                self.dataset.num_init_thetas,
                self.batch_size,
                replace=False
            )
        else:
            indices = np.arange(self.batch_size)

        # Select batch
        batch_inputs = inputs[indices]  # Shape: (batch_size, num_t, 4)
        batch_targets = targets[indices]  # Shape: (batch_size, num_t, 2)

        self.goal_idx += 1

        return batch_inputs, batch_targets

    def __len__(self) -> int:
        """Return number of batches per episode."""
        return self.dataset.num_init_thetas // self.batch_size


# Example usage:
if __name__ == "__main__":
    from arguments_for_runs import get_dataset_args, save_args

    args = get_dataset_args()

    # Create dataset
    dataset = PlanarArmDataset(
        num_t=args.num_t,
        wait_steps_after_trajectory=args.wait_steps,
        num_init_thetas=args.num_init_thetas,
        num_goals=args.num_goals,
        movement_duration=args.movement_duration,
        train_split=args.train_split,
        save_dir=args.save_dir,
    )

    # TODO: Add method in dataset to check if data match args. If not then re-generate dataset
    save_args(args, save_name=args.save_dir + '/dataset_args.txt')

    """
    # DEMO:
    
    dataset = PlanarArmDataset(
    num_t=100,
    num_init_thetas=1000,
    num_goals=50,
    train_split=0.8  # 80% training, 20% evaluation
    )

    train_loader = PlanarArmDataLoader(dataset, batch_size=32, mode='train')
    eval_loader = PlanarArmDataLoader(dataset, batch_size=32, mode='eval')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        for inputs, targets in train_loader:
            # Training step
            inputs.shape # (batch_size, num_t, 4)
            targets.shape # (batch_size, num_t, 2)
            
        # Evaluate
        for inputs, targets in eval_loader:
            # Evaluation step
            outputs = model(inputs)
            # Rescale
            pred_targets = dataset.inverse_transform_targets(outputs)
            
    """