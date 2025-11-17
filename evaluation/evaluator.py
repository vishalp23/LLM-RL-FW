"""
Evaluator for LLM-RL Framework.

Provides evaluation and comparison tools for agents.
"""

import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path


class Evaluator:
    """
    Evaluates agent performance over multiple episodes.

    Tracks:
    - Success rate
    - Mean reward
    - Mean episode length
    - Additional custom metrics
    """

    def __init__(self, env_wrapper, num_episodes: int = 10, verbose: bool = True):
        """
        Initialize evaluator.

        Args:
            env_wrapper: MiniGrid environment wrapper
            num_episodes: Number of episodes to evaluate (default: 10)
            verbose: Whether to print progress (default: True)
        """
        self.env = env_wrapper
        self.num_episodes = num_episodes
        self.verbose = verbose

    def evaluate_agent(self, agent, num_episodes: int = None) -> Dict[str, Any]:
        """
        Evaluate agent over multiple episodes.

        Args:
            agent: Agent to evaluate
            num_episodes: Number of episodes (default: self.num_episodes)

        Returns:
            Dictionary with evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        # Track metrics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_times = []

        if self.verbose:
            print(f"\nEvaluating {agent} for {num_episodes} episodes...")
            print("=" * 60)

        # Run episodes
        for episode in range(num_episodes):
            episode_start = time.time()

            # Reset environment and agent
            obs = self.env.reset()
            agent.reset()

            episode_reward = 0.0
            episode_length = 0
            done = False

            # Run episode
            while not done:
                # Agent selects action
                action, info = agent.step(obs)

                # Environment step
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated

                # Update agent
                agent.update(obs, action, reward, next_obs, done, step_info)

                # Track metrics
                episode_reward += reward
                episode_length += 1

                # Move to next observation
                obs = next_obs

            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(step_info.get('success', False))
            episode_times.append(time.time() - episode_start)

            # Print progress
            if self.verbose:
                success_str = "SUCCESS" if step_info.get('success', False) else "FAILED"
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, "
                      f"Status={success_str}, "
                      f"Time={episode_times[-1]:.2f}s")

        # Calculate aggregate metrics
        results = {
            'agent_type': str(agent),
            'num_episodes': num_episodes,
            'success_rate': np.mean(episode_successes),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_time_per_episode': np.mean(episode_times),
            'total_time': np.sum(episode_times),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
        }

        if self.verbose:
            print("=" * 60)
            print(f"Evaluation Complete:")
            print(f"  Success Rate: {results['success_rate']:.2%}")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            print(f"  Total Time: {results['total_time']:.1f}s")
            print("=" * 60)

        return results

    def compare_agents(self, agents: List[Any], agent_names: List[str] = None,
                       num_episodes: int = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents.

        Args:
            agents: List of agents to compare
            agent_names: List of names for agents (default: agent.__str__())
            num_episodes: Number of episodes per agent (default: self.num_episodes)

        Returns:
            Dictionary mapping agent names to evaluation results
        """
        if agent_names is None:
            agent_names = [str(agent) for agent in agents]

        if len(agent_names) != len(agents):
            raise ValueError("Number of agent names must match number of agents")

        results = {}

        for agent, name in zip(agents, agent_names):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Evaluating: {name}")
                print(f"{'=' * 60}")

            agent_results = self.evaluate_agent(agent, num_episodes)
            results[name] = agent_results

        return results

    def print_comparison(self, comparison_results: Dict[str, Dict[str, Any]]):
        """
        Print formatted comparison table.

        Args:
            comparison_results: Results from compare_agents()
        """
        print("\n" + "=" * 80)
        print("AGENT COMPARISON RESULTS")
        print("=" * 80)

        # Header
        print(f"{'Agent':<30} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15}")
        print("-" * 80)

        # Sort by success rate
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )

        # Print rows
        for agent_name, results in sorted_results:
            # Truncate long agent names
            display_name = agent_name[:28] + '..' if len(agent_name) > 30 else agent_name

            print(f"{display_name:<30} "
                  f"{results['success_rate']:>6.1%}        "
                  f"{results['mean_reward']:>6.2f} ± {results['std_reward']:<4.2f}  "
                  f"{results['mean_length']:>6.1f} ± {results['std_length']:<4.1f}")

        print("=" * 80)

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to output JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_serializable(results)

        # Ensure directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to: {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python types for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.

        Args:
            input_path: Path to input JSON file

        Returns:
            Evaluation results dictionary
        """
        with open(input_path, 'r') as f:
            results = json.load(f)

        if self.verbose:
            print(f"Results loaded from: {input_path}")

        return results


def quick_evaluate(agent, env_wrapper, num_episodes: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function for quick agent evaluation.

    Args:
        agent: Agent to evaluate
        env_wrapper: Environment wrapper
        num_episodes: Number of episodes
        verbose: Whether to print progress

    Returns:
        Evaluation results dictionary
    """
    evaluator = Evaluator(env_wrapper, num_episodes, verbose)
    return evaluator.evaluate_agent(agent)


def quick_compare(agents: List[Any], agent_names: List[str], env_wrapper,
                  num_episodes: int = 10, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function for quick agent comparison.

    Args:
        agents: List of agents to compare
        agent_names: List of agent names
        env_wrapper: Environment wrapper
        num_episodes: Number of episodes per agent
        verbose: Whether to print progress

    Returns:
        Comparison results dictionary
    """
    evaluator = Evaluator(env_wrapper, num_episodes, verbose)
    results = evaluator.compare_agents(agents, agent_names)
    evaluator.print_comparison(results)
    return results
