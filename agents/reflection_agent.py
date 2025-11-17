"""
Reflection Agent for LLM-RL Framework.

Extends MemoryAgent with self-reflection capabilities.
"""

from typing import Dict, List, Any
from agents.memory_agent import MemoryAgent


class ReflectionAgent(MemoryAgent):
    """
    Agent with self-reflection capabilities that learns from experience analysis.

    Extends MemoryAgent by:
    1. Generating reflections periodically by analyzing past episodes
    2. Comparing successful vs failed episodes to extract insights
    3. Maintaining a reflection buffer
    4. Augmenting prompts with both memory and reflection context
    """

    def __init__(self, llm_interface, env_wrapper, agent_config: Dict[str, Any] = None):
        """
        Initialize the reflection agent.

        Args:
            llm_interface: Interface to LLM
            env_wrapper: MiniGrid environment wrapper
            agent_config: Configuration dictionary
                - reflection_frequency: Episodes between reflections (default: 5)
                - max_reflections: Maximum reflections to store (default: 5)
                - All MemoryAgent configs also apply
        """
        super().__init__(llm_interface, env_wrapper, agent_config)

        # Reflection configuration
        self.reflection_frequency = self.config.get('reflection_frequency', 5)
        self.max_reflections = self.config.get('max_reflections', 5)

        # Reflection storage
        self.reflections = []
        self.last_reflection_episode = 0

    def _end_episode(self, info: Dict[str, Any]):
        """
        Handle end of episode and trigger reflection if needed.

        Args:
            info: Info dict from environment
        """
        # Call parent to store in memory
        super()._end_episode(info)

        # Check if it's time to reflect
        episodes_since_reflection = self.episode_count - self.last_reflection_episode

        if episodes_since_reflection >= self.reflection_frequency and len(self.memory) >= 2:
            self._generate_reflection()
            self.last_reflection_episode = self.episode_count

    def _generate_reflection(self):
        """
        Generate reflection by analyzing recent episodes.
        """
        # Get successful and failed episodes from memory
        successful = [ep for ep in self.memory if ep['success']]
        failed = [ep for ep in self.memory if not ep['success']]

        # Create reflection prompt
        reflection_prompt = self._create_reflection_prompt(successful, failed)

        # Get LLM reflection
        try:
            reflection_text = self.llm.generate(reflection_prompt)

            # Store reflection
            reflection = {
                'episode_num': self.episode_count,
                'text': reflection_text,
                'num_successful': len(successful),
                'num_failed': len(failed)
            }

            self.reflections.append(reflection)

            # Limit reflection buffer size
            if len(self.reflections) > self.max_reflections:
                self.reflections = self.reflections[-self.max_reflections:]

        except Exception as e:
            print(f"Warning: Failed to generate reflection: {e}")

    def _create_reflection_prompt(self, successful: List[Dict], failed: List[Dict]) -> str:
        """
        Create prompt for reflection generation.

        Args:
            successful: List of successful episodes
            failed: List of failed episodes

        Returns:
            Reflection prompt string
        """
        prompt = """You are analyzing your performance in a MiniGrid navigation task.
Based on your recent episodes, reflect on what strategies work and what doesn't.

"""

        # Add successful episode information
        if successful:
            prompt += f"=== Successful Episodes ({len(successful)}) ===\n"
            for i, ep in enumerate(successful[-3:], 1):  # Last 3 successful
                prompt += f"\nEpisode {ep['episode_num']} (Reward: {ep['reward']:.2f}, Steps: {ep['length']}):\n"
                actions = [step['action'] for step in ep['trajectory']]
                prompt += f"Actions: {' -> '.join(actions[:15])}\n"
                if len(actions) > 15:
                    prompt += "...\n"

        # Add failed episode information
        if failed:
            prompt += f"\n=== Failed Episodes ({len(failed)}) ===\n"
            for i, ep in enumerate(failed[-3:], 1):  # Last 3 failed
                prompt += f"\nEpisode {ep['episode_num']} (Reward: {ep['reward']:.2f}, Steps: {ep['length']}):\n"
                actions = [step['action'] for step in ep['trajectory']]
                prompt += f"Actions: {' -> '.join(actions[:15])}\n"
                if len(actions) > 15:
                    prompt += "...\n"

        prompt += """
Please provide 2-3 key insights about:
1. What strategies or action patterns led to success?
2. What mistakes or patterns led to failure?
3. What should you focus on improving?

Keep your reflection concise and actionable (3-5 sentences).

Reflection: """

        return prompt

    def _format_reflection_context(self) -> str:
        """
        Format reflection context for prompt augmentation.

        Returns:
            Formatted string with reflection insights
        """
        if not self.reflections:
            return ""

        context = "\n=== Self-Reflection Insights ===\n"
        context += "Based on analyzing past performance, here are key insights:\n\n"

        # Include most recent reflections (up to 2)
        recent_reflections = self.reflections[-2:]

        for i, reflection in enumerate(recent_reflections, 1):
            context += f"Insight {i}:\n{reflection['text']}\n\n"

        return context

    def create_prompt(self, obs: Dict[str, Any], available_actions: List[str]) -> str:
        """
        Create prompt augmented with both memory and reflection context.

        Args:
            obs: Current observation
            available_actions: List of available action names

        Returns:
            Formatted prompt string with memory and reflection context
        """
        # Start with base observation formatting
        prompt = f"""You are an AI agent playing a MiniGrid game. Your goal is to navigate the grid and reach the objective.

{self.format_observation(obs)}
"""

        # Add reflection context
        reflection_context = self._format_reflection_context()
        if reflection_context:
            prompt += reflection_context

        # Add memory context (successful examples)
        memory_context = self._format_memory_context()
        if memory_context:
            prompt += memory_context

        # Add available actions
        prompt += "\nAvailable Actions:\n"
        for idx, action in enumerate(available_actions):
            prompt += f"{idx}: {action}\n"

        # Add final instruction
        prompt += """
Please analyze the situation and choose the best action. Respond ONLY with the action number (0-6).

Your response should be just a single number, nothing else.

Action: """

        return prompt

    def get_reflections(self) -> List[Dict[str, Any]]:
        """
        Get all stored reflections.

        Returns:
            List of reflection dictionaries
        """
        return self.reflections.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics including reflection information.

        Returns:
            Dictionary with agent statistics
        """
        stats = super().get_statistics()
        stats['agent_type'] = 'ReflectionAgent'
        stats['num_reflections'] = len(self.reflections)
        stats['reflection_frequency'] = self.reflection_frequency
        return stats

    def __str__(self) -> str:
        """String representation of agent."""
        mem_summary = self.get_memory_summary()
        return (f"ReflectionAgent(episodes={self.episode_count}, "
                f"success_rate={self.successful_episodes/max(1, self.episode_count):.2%}, "
                f"memory={mem_summary['successful_episodes']}/{len(self.memory)} successful, "
                f"reflections={len(self.reflections)})")
