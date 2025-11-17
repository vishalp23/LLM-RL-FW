"""
Base Agent for LLM-RL Framework.

Uses structured prompting to convert MiniGrid observations to actions.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json


class BaseAgent:
    """
    Base agent that uses LLM for decision-making in MiniGrid environments.

    Uses structured prompting to convert observations to actions by:
    1. Formatting grid observations in human-readable format
    2. Providing context about available actions
    3. Parsing LLM responses to extract action indices
    4. Tracking episode history for statistics
    """

    def __init__(self, llm_interface, env_wrapper, agent_config: Dict[str, Any] = None):
        """
        Initialize the base agent.

        Args:
            llm_interface: Interface to LLM (e.g., OllamaInterface)
            env_wrapper: MiniGrid environment wrapper
            agent_config: Configuration dictionary for agent parameters
        """
        self.llm = llm_interface
        self.env = env_wrapper
        self.config = agent_config or {}

        # Episode tracking
        self.current_episode_history = []
        self.all_episodes = []
        self.episode_count = 0

        # Statistics
        self.total_steps = 0
        self.total_rewards = 0.0
        self.successful_episodes = 0

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """
        Format MiniGrid observation into human-readable text.

        Args:
            obs: Observation dictionary from environment

        Returns:
            Formatted string representation of observation
        """
        formatted = "=== Current Observation ===\n"

        # Grid representation
        if 'grid' in obs:
            grid = obs['grid']
            formatted += f"Grid (shape {grid.shape}):\n"
            formatted += self._format_grid(grid)
            formatted += "\n"

        # Direction
        if 'direction' in obs:
            directions = ['right', 'down', 'left', 'up']
            dir_idx = obs['direction']
            formatted += f"Agent Direction: {directions[dir_idx]}\n"

        # Mission
        if 'mission' in obs:
            formatted += f"Mission: {obs['mission']}\n"

        return formatted

    def _format_grid(self, grid: np.ndarray) -> str:
        """
        Format grid array into readable text representation.

        Args:
            grid: Grid array from MiniGrid

        Returns:
            String representation of grid
        """
        # MiniGrid encoding: each cell is [object_type, color, state]
        object_types = {
            0: '.',   # Empty
            1: 'W',   # Wall
            2: 'D',   # Door
            3: 'K',   # Key
            4: 'B',   # Ball
            5: 'X',   # Box
            6: 'G',   # Goal
            7: 'L',   # Lava
            8: 'A',   # Agent
        }

        rows, cols = grid.shape[:2]
        grid_str = ""

        for i in range(rows):
            row_str = ""
            for j in range(cols):
                obj_type = grid[i, j, 0]
                symbol = object_types.get(obj_type, '?')
                row_str += symbol + " "
            grid_str += row_str.strip() + "\n"

        return grid_str

    def create_prompt(self, obs: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Create structured prompt for LLM to generate action.

        Args:
            obs: Current observation
            available_actions: List of available action names

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an AI agent playing a MiniGrid game. Your goal is to navigate the grid and reach the objective.

{self.format_observation(obs)}

Available Actions:
"""
        for idx, action in enumerate(available_actions):
            prompt += f"{idx}: {action}\n"

        prompt += """
Please analyze the situation and choose the best action. Respond ONLY with the action number (0-6).

Your response should be just a single number, nothing else.

Action: """

        return prompt

    def parse_action(self, llm_response: str, num_actions: int) -> int:
        """
        Parse LLM response to extract action index.

        Args:
            llm_response: Raw response from LLM
            num_actions: Number of available actions

        Returns:
            Action index (0 to num_actions-1)
        """
        # Try to find a number in the response
        import re

        # Remove whitespace
        response = llm_response.strip()

        # Try to extract first number
        numbers = re.findall(r'\d+', response)

        if numbers:
            action = int(numbers[0])
            # Validate action is in valid range
            if 0 <= action < num_actions:
                return action

        # Default to action 0 if parsing fails
        print(f"Warning: Could not parse action from response: '{llm_response}'. Defaulting to 0.")
        return 0

    def select_action(self, obs: Dict[str, Any]) -> int:
        """
        Select action based on observation using LLM.

        Args:
            obs: Current observation from environment

        Returns:
            Action index
        """
        # Get available actions
        available_actions = self.env.get_available_actions()

        # Create prompt
        prompt = self.create_prompt(obs, available_actions)

        # Get LLM response
        llm_response = self.llm.generate(prompt)

        # Parse action
        action = self.parse_action(llm_response, len(available_actions))

        return action

    def step(self, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            obs: Current observation

        Returns:
            Tuple of (action, info_dict)
        """
        action = self.select_action(obs)

        info = {
            'agent_type': 'BaseAgent',
            'action': action
        }

        return action, info

    def update(self, obs: Dict[str, Any], action: int, reward: float,
               next_obs: Dict[str, Any], done: bool, info: Dict[str, Any]):
        """
        Update agent after environment step.

        Args:
            obs: Previous observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode is done
            info: Additional info from environment
        """
        # Track step in episode history
        step_data = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'info': info
        }
        self.current_episode_history.append(step_data)

        # Update statistics
        self.total_steps += 1
        self.total_rewards += reward

        # End of episode
        if done:
            self._end_episode(info)

    def _end_episode(self, info: Dict[str, Any]):
        """
        Handle end of episode bookkeeping.

        Args:
            info: Info dict from environment
        """
        # Calculate episode statistics
        episode_length = len(self.current_episode_history)
        episode_reward = sum(step['reward'] for step in self.current_episode_history)
        success = info.get('success', False)

        # Store episode
        episode_data = {
            'episode_num': self.episode_count,
            'history': self.current_episode_history,
            'length': episode_length,
            'reward': episode_reward,
            'success': success
        }
        self.all_episodes.append(episode_data)

        # Update counters
        self.episode_count += 1
        if success:
            self.successful_episodes += 1

        # Reset current episode
        self.current_episode_history = []

    def reset(self):
        """Reset agent for new episode."""
        self.current_episode_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        stats = {
            'agent_type': 'BaseAgent',
            'total_episodes': self.episode_count,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.episode_count),
            'total_steps': self.total_steps,
            'total_rewards': self.total_rewards,
            'avg_steps_per_episode': self.total_steps / max(1, self.episode_count),
            'avg_reward_per_episode': self.total_rewards / max(1, self.episode_count),
        }

        return stats

    def __str__(self) -> str:
        """String representation of agent."""
        return f"BaseAgent(episodes={self.episode_count}, success_rate={self.successful_episodes/max(1, self.episode_count):.2%})"
