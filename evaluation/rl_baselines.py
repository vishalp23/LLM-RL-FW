"""
RL Baselines for LLM-RL Framework.

Provides baseline RL agents for comparison with LLM agents.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings

# Suppress stable-baselines3 warnings
warnings.filterwarnings('ignore')

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. RL baselines will not work.")


class RandomAgent:
    """
    Random baseline agent.

    Selects actions uniformly at random from available actions.
    Provides consistent interface with LLM agents for fair comparison.
    """

    def __init__(self, env_wrapper, agent_config: Dict[str, Any] = None):
        """
        Initialize random agent.

        Args:
            env_wrapper: MiniGrid environment wrapper
            agent_config: Configuration (unused, for compatibility)
        """
        self.env = env_wrapper
        self.config = agent_config or {}

        # Episode tracking (for consistency with LLM agents)
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = 0.0
        self.successful_episodes = 0

    def step(self, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Select random action.

        Args:
            obs: Current observation (unused)

        Returns:
            Tuple of (action, info_dict)
        """
        # Get number of actions
        num_actions = len(self.env.get_available_actions())

        # Select random action
        action = np.random.randint(0, num_actions)

        info = {
            'agent_type': 'RandomAgent',
            'action': action
        }

        return action, info

    def update(self, obs: Dict[str, Any], action: int, reward: float,
               next_obs: Dict[str, Any], done: bool, info: Dict[str, Any]):
        """
        Update agent (no learning for random agent).

        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
            info: Additional info
        """
        # Update statistics
        self.total_steps += 1
        self.total_rewards += reward

        if done:
            self.episode_count += 1
            if info.get('success', False):
                self.successful_episodes += 1

    def reset(self):
        """Reset agent for new episode."""
        pass  # No state to reset for random agent

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_type': 'RandomAgent',
            'total_episodes': self.episode_count,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.episode_count),
            'total_steps': self.total_steps,
            'total_rewards': self.total_rewards,
            'avg_steps_per_episode': self.total_steps / max(1, self.episode_count),
            'avg_reward_per_episode': self.total_rewards / max(1, self.episode_count),
        }

    def __str__(self) -> str:
        """String representation."""
        return "RandomAgent"


class RLBaselineAgent:
    """
    Wrapper for Stable Baselines3 RL agents.

    Provides consistent interface with LLM agents for fair comparison.
    Supports PPO, DQN, and A2C algorithms.
    """

    def __init__(self, env_wrapper, algorithm: str = 'PPO', agent_config: Dict[str, Any] = None):
        """
        Initialize RL baseline agent.

        Args:
            env_wrapper: MiniGrid environment wrapper
            algorithm: RL algorithm to use ('PPO', 'DQN', or 'A2C')
            agent_config: Configuration dictionary
                - learning_rate: Learning rate (default: 3e-4)
                - total_timesteps: Training timesteps (default: 100000)
                - verbose: Verbosity level (default: 0)
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RL baselines. "
                              "Install with: pip install stable-baselines3")

        self.env = env_wrapper
        self.algorithm_name = algorithm.upper()
        self.config = agent_config or {}

        # Get config parameters
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.total_timesteps = self.config.get('total_timesteps', 100000)
        self.verbose = self.config.get('verbose', 0)

        # Create vectorized environment for stable-baselines3
        self.vec_env = DummyVecEnv([lambda: env_wrapper.env])

        # Initialize RL model
        self.model = self._create_model()

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = 0.0
        self.successful_episodes = 0
        self.is_trained = False

    def _create_model(self):
        """
        Create RL model based on algorithm.

        Returns:
            Stable Baselines3 model instance
        """
        model_kwargs = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': self.learning_rate,
            'verbose': self.verbose,
        }

        if self.algorithm_name == 'PPO':
            return PPO(**model_kwargs)
        elif self.algorithm_name == 'DQN':
            return DQN(**model_kwargs)
        elif self.algorithm_name == 'A2C':
            return A2C(**model_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}. "
                             f"Must be one of: PPO, DQN, A2C")

    def train(self, total_timesteps: int = None, verbose: bool = True):
        """
        Train the RL agent.

        Args:
            total_timesteps: Number of training timesteps (default: self.total_timesteps)
            verbose: Whether to print training progress
        """
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        if verbose:
            print(f"\nTraining {self.algorithm_name} agent for {total_timesteps} timesteps...")

        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True

        if verbose:
            print(f"{self.algorithm_name} training complete!")

    def step(self, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using trained policy.

        Args:
            obs: Current observation

        Returns:
            Tuple of (action, info_dict)
        """
        if not self.is_trained:
            print(f"Warning: {self.algorithm_name} agent not trained. "
                  f"Call train() before evaluation.")

        # Convert observation to format expected by SB3
        # For MiniGrid, we need the 'image' observation
        sb3_obs = obs['grid'].flatten()

        # Get action from model
        action, _states = self.model.predict(sb3_obs, deterministic=True)

        # Convert to int (SB3 returns numpy array)
        action = int(action)

        info = {
            'agent_type': f'{self.algorithm_name}Agent',
            'action': action
        }

        return action, info

    def update(self, obs: Dict[str, Any], action: int, reward: float,
               next_obs: Dict[str, Any], done: bool, info: Dict[str, Any]):
        """
        Update agent statistics (no online learning).

        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
            info: Additional info
        """
        # Update statistics
        self.total_steps += 1
        self.total_rewards += reward

        if done:
            self.episode_count += 1
            if info.get('success', False):
                self.successful_episodes += 1

    def reset(self):
        """Reset agent for new episode."""
        pass  # No episode-level state to reset

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            'agent_type': f'{self.algorithm_name}Agent',
            'total_episodes': self.episode_count,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.episode_count),
            'total_steps': self.total_steps,
            'total_rewards': self.total_rewards,
            'avg_steps_per_episode': self.total_steps / max(1, self.episode_count),
            'avg_reward_per_episode': self.total_rewards / max(1, self.episode_count),
            'is_trained': self.is_trained,
        }

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        print(f"Model saved to: {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        if self.algorithm_name == 'PPO':
            self.model = PPO.load(path, env=self.vec_env)
        elif self.algorithm_name == 'DQN':
            self.model = DQN.load(path, env=self.vec_env)
        elif self.algorithm_name == 'A2C':
            self.model = A2C.load(path, env=self.vec_env)

        self.is_trained = True
        print(f"Model loaded from: {path}")

    def __str__(self) -> str:
        """String representation."""
        trained_str = "trained" if self.is_trained else "untrained"
        return f"{self.algorithm_name}Agent({trained_str})"


def create_baseline_agents(env_wrapper, train: bool = True,
                           train_timesteps: int = 100000) -> Dict[str, Any]:
    """
    Create all baseline agents for comparison.

    Args:
        env_wrapper: MiniGrid environment wrapper
        train: Whether to train RL agents (default: True)
        train_timesteps: Training timesteps for RL agents (default: 100000)

    Returns:
        Dictionary mapping agent names to agent instances
    """
    agents = {}

    # Random agent
    agents['Random'] = RandomAgent(env_wrapper)

    # RL agents (if available)
    if SB3_AVAILABLE:
        for algorithm in ['PPO', 'DQN', 'A2C']:
            agent = RLBaselineAgent(env_wrapper, algorithm=algorithm)
            if train:
                agent.train(total_timesteps=train_timesteps)
            agents[algorithm] = agent
    else:
        print("Warning: Stable Baselines3 not available. Only Random agent created.")

    return agents
