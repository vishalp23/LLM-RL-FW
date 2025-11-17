"""
MiniGrid Environment Wrapper for LLM-RL Framework.

Provides LLM-friendly interface to MiniGrid environments.
"""

import gymnasium as gym
import minigrid  # This registers the environments with gymnasium
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MiniGridWrapper:
    """
    Wrapper for MiniGrid environments with LLM-friendly interface.

    Provides:
    1. Human-readable action names
    2. Formatted observations
    3. Success/failure tracking
    4. Consistent interface for LLM agents
    """

    # MiniGrid action mappings
    ACTION_NAMES = [
        'turn_left',
        'turn_right',
        'move_forward',
        'pickup',
        'drop',
        'toggle',
        'done'
    ]

    def __init__(self, env_name: str, max_steps: int = None, render_mode: str = None):
        """
        Initialize MiniGrid wrapper.

        Args:
            env_name: Name of MiniGrid environment (e.g., "MiniGrid-Empty-8x8-v0")
            max_steps: Maximum steps per episode (default: env default)
            render_mode: Render mode for visualization (default: None)
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Create environment
        self.env = gym.make(env_name, render_mode=render_mode)

        # Override max steps if specified
        if max_steps is not None:
            self.env.unwrapped.max_steps = max_steps

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment for new episode.

        Returns:
            Initial observation dictionary
        """
        obs, info = self.env.reset()

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # Format observation
        formatted_obs = self._format_observation(obs, info)

        return formatted_obs

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take step in environment.

        Args:
            action: Action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update tracking
        self.current_step += 1
        self.episode_reward += reward

        # Add success flag to info
        done = terminated or truncated
        if done:
            self.episode_count += 1
            # MiniGrid provides success in reward (1.0 for reaching goal)
            info['success'] = reward > 0

        # Format observation
        formatted_obs = self._format_observation(obs, info)

        return formatted_obs, reward, terminated, truncated, info

    def _format_observation(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format observation into LLM-friendly dictionary.

        Args:
            obs: Raw observation from environment
            info: Info dict from environment

        Returns:
            Formatted observation dictionary
        """
        # MiniGrid observation structure: {image: ndarray, direction: int, mission: str}
        if isinstance(obs, dict):
            # Already formatted (newer MiniGrid versions)
            formatted = {
                'grid': obs.get('image', np.array([])),
                'direction': obs.get('direction', 0),
                'mission': obs.get('mission', ''),
            }
        else:
            # Handle older format or wrapped observations
            formatted = {
                'grid': obs,
                'direction': 0,  # Default direction
                'mission': '',
            }

        # Add metadata
        formatted['step'] = self.current_step
        formatted['episode_reward'] = self.episode_reward

        return formatted

    def get_available_actions(self) -> List[str]:
        """
        Get list of available action names.

        Returns:
            List of action name strings
        """
        return self.ACTION_NAMES.copy()

    def get_action_name(self, action: int) -> str:
        """
        Get name of action from index.

        Args:
            action: Action index

        Returns:
            Action name string
        """
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"unknown_{action}"

    def get_action_index(self, action_name: str) -> int:
        """
        Get index of action from name.

        Args:
            action_name: Action name string

        Returns:
            Action index
        """
        try:
            return self.ACTION_NAMES.index(action_name)
        except ValueError:
            return 0  # Default to turn_left

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment."""
        self.env.close()

    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information.

        Returns:
            Dictionary with environment info
        """
        return {
            'env_name': self.env_name,
            'max_steps': self.env.unwrapped.max_steps,
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'episode_count': self.episode_count,
            'num_actions': len(self.ACTION_NAMES),
        }

    def __str__(self) -> str:
        """String representation."""
        return f"MiniGridWrapper({self.env_name})"


class MiniGridTaskSuite:
    """
    Collection of MiniGrid tasks organized by difficulty.

    Provides easy access to task sets for evaluation:
    - Easy: Simple navigation tasks
    - Medium: Tasks with obstacles or basic interactions
    - Hard: Complex tasks requiring planning
    """

    # Task definitions
    EASY_TASKS = [
        'MiniGrid-Empty-5x5-v0',
        'MiniGrid-Empty-6x6-v0',
        'MiniGrid-Empty-8x8-v0',
    ]

    MEDIUM_TASKS = [
        'MiniGrid-FourRooms-v0',
        'MiniGrid-DoorKey-5x5-v0',
        'MiniGrid-DoorKey-6x6-v0',
        'MiniGrid-SimpleCrossingS9N1-v0',
    ]

    HARD_TASKS = [
        'MiniGrid-DoorKey-8x8-v0',
        'MiniGrid-MultiRoom-N2-S4-v0',
        'MiniGrid-Unlock-v0',
        'MiniGrid-UnlockPickup-v0',
    ]

    @classmethod
    def get_easy_tasks(cls) -> List[str]:
        """Get list of easy task names."""
        return cls.EASY_TASKS.copy()

    @classmethod
    def get_medium_tasks(cls) -> List[str]:
        """Get list of medium task names."""
        return cls.MEDIUM_TASKS.copy()

    @classmethod
    def get_hard_tasks(cls) -> List[str]:
        """Get list of hard task names."""
        return cls.HARD_TASKS.copy()

    @classmethod
    def get_all_tasks(cls) -> List[str]:
        """Get all tasks across all difficulty levels."""
        return cls.EASY_TASKS + cls.MEDIUM_TASKS + cls.HARD_TASKS

    @classmethod
    def get_tasks_by_difficulty(cls, difficulty: str) -> List[str]:
        """
        Get tasks for specified difficulty level.

        Args:
            difficulty: One of 'easy', 'medium', 'hard', or 'all'

        Returns:
            List of task names

        Raises:
            ValueError: If difficulty is not recognized
        """
        difficulty = difficulty.lower()

        if difficulty == 'easy':
            return cls.get_easy_tasks()
        elif difficulty == 'medium':
            return cls.get_medium_tasks()
        elif difficulty == 'hard':
            return cls.get_hard_tasks()
        elif difficulty == 'all':
            return cls.get_all_tasks()
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}. "
                             f"Must be one of: easy, medium, hard, all")

    @classmethod
    def create_wrapper(cls, task_name: str, **kwargs) -> MiniGridWrapper:
        """
        Create MiniGrid wrapper for specified task.

        Args:
            task_name: Name of task
            **kwargs: Additional arguments for wrapper

        Returns:
            MiniGridWrapper instance
        """
        return MiniGridWrapper(task_name, **kwargs)

    @classmethod
    def get_task_info(cls) -> Dict[str, Any]:
        """
        Get information about available tasks.

        Returns:
            Dictionary with task statistics
        """
        return {
            'easy_tasks': len(cls.EASY_TASKS),
            'medium_tasks': len(cls.MEDIUM_TASKS),
            'hard_tasks': len(cls.HARD_TASKS),
            'total_tasks': len(cls.get_all_tasks()),
            'easy_task_list': cls.EASY_TASKS,
            'medium_task_list': cls.MEDIUM_TASKS,
            'hard_task_list': cls.HARD_TASKS,
        }
