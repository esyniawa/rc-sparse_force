import numpy as np
import torch
from typing import Optional, Dict
from tqdm import tqdm
import os
from kinematics.environment import ReachingEnvironment
from networks.rl_sparce import RLSpaRCe
from networks.utils import action_no_grad
from plotting_functions import analyze_performance
import warnings

# Yeah there are warnings. TODO: check them all
warnings.filterwarnings("ignore")


def anneal_decay(epsilon: float, final_epsilon: float, epoch: int, num_epochs: int) -> float:
    return final_epsilon + (epsilon - final_epsilon) * np.exp(-5.0 * epoch / num_epochs)


def train_rl_sparce(
        model: RLSpaRCe,
        num_epochs: int,
        num_episodes: int,
        max_steps: int,
        device: torch.device,
        result_folder: str,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        test_interval: int = 1,
        seed: Optional[int] = None
) -> Dict[str, list]:
    """
    Train RLSpaRCe model on reaching task with evaluation.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(result_folder, exist_ok=True)

    # Training history
    history = {
        'train_rewards': [],
        'train_steps': [],
        'train_success_rate': [],
        'test_results': []
    }

    # Create multiple environments for batch processing
    batch_size = 32  # Adjust based on your needs
    envs = [ReachingEnvironment(init_thetas=np.zeros(2)) for _ in range(batch_size)]

    # Initialize traces + percentile thresholds using random actions
    model.initialize_traces(batch_size)
    model.compute_initial_percentiles(envs)

    # Training loop over epochs
    for epoch in range(num_epochs):
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Calculate current epsilon
        epsilon = anneal_decay(initial_epsilon, final_epsilon, epoch, num_epochs)

        # Training phase
        train_rewards = []
        train_steps = []
        train_successes = 0

        with torch.no_grad():
            for _ in tqdm(range(num_episodes), desc="Training"):
                # Reset environments and model for new episode
                states = torch.from_numpy(np.array([env.reset() for env in envs])).float().to(device)
                model.reset_state(batch_size)
                model.reset_traces()

                episode_rewards = torch.zeros(batch_size).to(device)
                episode_steps = torch.zeros(batch_size).to(device)
                active_envs = torch.ones(batch_size, dtype=torch.bool).to(device)

                for step in range(max_steps):
                    # Select actions using epsilon-greedy policy
                    actions = model.train_step(states, epsilon=epsilon)

                    # Step environments
                    next_states = []
                    rewards = []
                    dones = []

                    for i, (env, action) in enumerate(zip(envs, actions)):
                        if active_envs[i]:
                            next_state, reward, done = env.step(action.cpu().numpy())
                            next_states.append(next_state)
                            rewards.append(reward)
                            dones.append(done)

                            episode_rewards[i] += reward
                            episode_steps[i] += 1

                            if done:
                                train_successes += 1
                                active_envs[i] = False
                        else:
                            next_states.append(np.zeros_like(states[0].cpu().numpy()))
                            rewards.append(0)
                            dones.append(True)

                    # Convert to tensors
                    next_states = torch.tensor(next_states).float().to(device)
                    rewards = torch.tensor(rewards).float().to(device)
                    dones = torch.tensor(dones).bool().to(device)

                    # Update model
                    model.update(rewards, next_states, dones)

                    # Update current states
                    states = next_states

                    # Break if all environments are done
                    if not active_envs.any():
                        break

                # Record episode statistics
                train_rewards.extend(episode_rewards.cpu().numpy())
                train_steps.extend(episode_steps.cpu().numpy())

        # Record training metrics
        avg_reward = np.mean(train_rewards)
        avg_steps = np.mean(train_steps)
        success_rate = (train_successes / (num_episodes * batch_size)) * 100

        history['train_rewards'].append(avg_reward)
        history['train_steps'].append(avg_steps)
        history['train_success_rate'].append(success_rate)

        # Testing phase
        if (epoch + 1) % test_interval == 0:
            test_results = evaluate_rl_sparce(
                model=model,
                num_episodes=num_episodes // 5,  # Fewer episodes for testing
                max_steps=max_steps,
                device=device
            )
            history['test_results'].append(test_results)

            # Plot and analyze results
            save_path = os.path.join(result_folder, f'epoch_{epoch + 1}_performance.pdf')
            analyze_performance(test_results, save_path=save_path)

            # Save model
            model_path = os.path.join(result_folder, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_path)

    return history


def evaluate_rl_sparce(
        model: RLSpaRCe,
        num_episodes: int,
        max_steps: int,
        device: torch.device
) -> Dict[str, list]:
    """
    Evaluate RLSpaRCe model on reaching task.
    """
    model.eval()

    # Create multiple environments for batch processing
    batch_size = 32
    envs = [ReachingEnvironment(init_thetas=np.zeros(2)) for _ in range(batch_size)]

    results = {
        'error': [],
        'steps': [],
        'total_reward': [],
        'success_rate': 0.0
    }

    num_batches = (num_episodes + batch_size - 1) // batch_size
    successes = 0

    with torch.no_grad():
        for _ in range(num_batches):
            # Reset environments and model
            states = torch.from_numpy(np.array([env.reset() for env in envs])).float().to(device)
            model.reset_state(batch_size)

            episode_rewards = np.zeros(batch_size)
            episode_steps = np.zeros(batch_size)
            final_errors = np.zeros(batch_size)
            active_envs = np.ones(batch_size, dtype=np.bool_)

            for step in range(max_steps):
                # Select actions (no exploration during evaluation)
                actions = model.test_step(states)

                # Step environments per batch
                next_states = []
                rewards = []
                dones = []

                for i, (env, action) in enumerate(zip(envs, actions)):
                    if active_envs[i]:
                        next_state, reward, done = env.step(action.cpu().numpy())
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)

                        episode_rewards[i] += reward
                        episode_steps[i] += 1
                        # Calculate final error
                        distance = env.target_pos - env.current_pos
                        final_errors[i] = np.linalg.norm(distance)

                        if done:
                            successes += 1
                            active_envs[i] = False
                    else:
                        next_states.append(np.zeros_like(states[0].cpu().numpy()))
                        rewards.append(0)
                        dones.append(True)

                # Convert to tensors
                states = torch.tensor(next_states).float().to(device)

                # Break if all environments are done
                if not active_envs.any():
                    break

            # Record batch results
            results['error'].extend(final_errors)
            results['steps'].extend(episode_steps)
            results['total_reward'].extend(episode_rewards)

    # Calculate success rate
    total_episodes = num_batches * batch_size
    results['success_rate'] = (successes / total_episodes) * 100.0

    return results


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = RLSpaRCe(
        dim_reservoir=1000,
        dim_input=6,  # current angles (2) + target position (2)
        dim_output=2,  # joint angle changes
        alpha=0.1,
        noise_scaling=0.0,
        feedforward_scaling=0.1,
        spectral_radius=1.0,
        percentile_n=10.0,
        probability_recurrent_connection=0.2,
        learning_rate_threshold=5e-6,
        learning_rate_readout=1e-5,
        device=device
    ).to(device)

    # Training parameters
    params = {
        'num_epochs': 20,
        'num_episodes': 100,
        'final_epsilon': 0.01,
        'max_steps': 200,
        'device': device,
        'result_folder': './results/rl_sparce_reaching',
        'seed': 42
    }

    # Train model
    history = train_rl_sparce(model, **params)
