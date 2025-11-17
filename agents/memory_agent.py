"""
Memory Agent for LLM-RL Framework.

Extends BaseAgent with episodic memory capabilities.
"""

from collections import deque
from typing import Dict, List, Any
from agents.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    """
    Agent with episodic memory that learns from past experiences.

    Extends BaseAgent by:
    1. Maintaining a memory buffer of past episodes
    2. Storing success/failure labels for each episode
    3. Augmenting prompts with successful episode examples
    4. Using configurable memory size and success threshold
    """

    def __init__(self, llm_interface, env_wrapper, agent_config: Dict[str, Any] = None):
        """
        Initialize the memory agent.

        Args:
            llm_interface: Interface to LLM
            env_wrapper: MiniGrid environment wrapper
            agent_config: Configuration dictionary
                - memory_size: Maximum number of episodes to store (default: 10)
                - success_threshold: Minimum reward to consider episode successful (default: 0.5)
                - num_examples: Number of successful examples to include in prompt (default: 2)
        """
        super().__init__(llm_interface, env_wrapper, agent_config)

        # Memory configuration
        self.memory_size = self.config.get('memory_size', 10)
        self.success_threshold = self.config.get('success_threshold', 0.5)
        self.num_examples = self.config.get('num_examples', 2)

        # Episodic memory (deque for efficient FIFO)
        self.memory = deque(maxlen=self.memory_size)

    def _end_episode(self, info: Dict[str, Any]):
        """
        Handle end of episode and store in memory.

        Args:
            info: Info dict from environment
        """
        # Calculate episode statistics
        episode_length = len(self.current_episode_history)
        episode_reward = sum(step['reward'] for step in self.current_episode_history)
        success = info.get('success', False) or (episode_reward >= self.success_threshold)

        # Create summarized episode data for memory
        # We store a compact representation to avoid memory bloat
        episode_summary = {
            'episode_num': self.episode_count,
            'length': episode_length,
            'reward': episode_reward,
            'success': success,
            'trajectory': self._summarize_trajectory(self.current_episode_history)
        }

        # Add to memory
        self.memory.append(episode_summary)

        # Call parent class method to handle statistics
        super()._end_episode(info)

    def _summarize_trajectory(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a compact summary of episode trajectory.

        Args:
            history: Full episode history

        Returns:
            Summarized trajectory
        """
        summary = []
        for step in history:
            # Get action name
            action_idx = step['action']
            action_name = self.env.get_available_actions()[action_idx]

            summary.append({
                'action': action_name,
                'reward': step['reward'],
            })

        return summary

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of current memory state.

        Returns:
            Dictionary with memory statistics
        """
        if len(self.memory) == 0:
            return {
                'total_episodes': 0,
                'successful_episodes': 0,
                'failed_episodes': 0,
                'success_rate': 0.0,
                'avg_successful_length': 0.0,
                'avg_failed_length': 0.0,
            }

        successful = [ep for ep in self.memory if ep['success']]
        failed = [ep for ep in self.memory if not ep['success']]

        summary = {
            'total_episodes': len(self.memory),
            'successful_episodes': len(successful),
            'failed_episodes': len(failed),
            'success_rate': len(successful) / len(self.memory),
            'avg_successful_length': sum(ep['length'] for ep in successful) / max(1, len(successful)),
            'avg_failed_length': sum(ep['length'] for ep in failed) / max(1, len(failed)),
        }

        return summary

    def _get_successful_examples(self, num_examples: int = None) -> List[Dict[str, Any]]:
        """
        Get successful episodes from memory.

        Args:
            num_examples: Number of examples to retrieve (default: self.num_examples)

        Returns:
            List of successful episode summaries
        """
        if num_examples is None:
            num_examples = self.num_examples

        # Filter successful episodes
        successful = [ep for ep in self.memory if ep['success']]

        # Return most recent successful episodes
        return list(successful)[-num_examples:]

    def _format_memory_context(self) -> str:
        """
        Format memory context for prompt augmentation.

        Returns:
            Formatted string with successful episode examples
        """
        examples = self._get_successful_examples()

        if not examples:
            return ""

        context = "\n=== Previous Successful Episodes ===\n"
        context += "Here are some examples of successful strategies from past episodes:\n\n"

        for i, episode in enumerate(examples, 1):
            context += f"Example {i} (Reward: {episode['reward']:.2f}, Steps: {episode['length']}):\n"
            context += "Actions taken: "

            # Format trajectory
            actions = [step['action'] for step in episode['trajectory']]
            context += " -> ".join(actions[:10])  # Limit to first 10 actions
            if len(actions) > 10:
                context += " -> ..."

            context += "\n\n"

        return context

    def create_prompt(self, obs: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Create prompt augmented with memory context.

        Args:
            obs: Current observation
            available_actions: List of available action names

        Returns:
            Formatted prompt string with memory context
        """
        # Get base prompt
        base_prompt = super().create_prompt(obs, available_actions)

        # Add memory context
        memory_context = self._format_memory_context()

        if memory_context:
            # Insert memory context before the final instruction
            prompt_parts = base_prompt.split("Please analyze")
            augmented_prompt = prompt_parts[0] + memory_context + "\nPlease analyze" + prompt_parts[1]
            return augmented_prompt
        else:
            return base_prompt

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics including memory information.

        Returns:
            Dictionary with agent statistics
        """
        stats = super().get_statistics()
        stats['agent_type'] = 'MemoryAgent'
        stats['memory_summary'] = self.get_memory_summary()
        return stats

    def __str__(self) -> str:
        """String representation of agent."""
        mem_summary = self.get_memory_summary()
        return (f"MemoryAgent(episodes={self.episode_count}, "
                f"success_rate={self.successful_episodes/max(1, self.episode_count):.2%}, "
                f"memory={mem_summary['successful_episodes']}/{len(self.memory)} successful)")
