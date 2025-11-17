"""
Example: Run BaseAgent on MiniGrid task.

Simple demonstration of BaseAgent with LLM decision-making.
"""

import sys
sys.path.append('..')

from agents.base_agent import BaseAgent
from models.ollama_interface import OllamaInterface, MockLLMInterface
from environments.minigrid_env import MiniGridWrapper
from evaluation.evaluator import quick_evaluate


def main():
    """Run BaseAgent example."""
    print("=" * 80)
    print("BaseAgent Example - LLM-RL Framework")
    print("=" * 80)

    # Configuration
    env_name = "MiniGrid-Empty-5x5-v0"  # Start with a simpler 5x5 grid
    model_name = "llama3.2"
    num_episodes = 5
    use_mock = True  # Use mock LLM for testing environment setup

    # Create environment
    print(f"\nInitializing environment: {env_name}")
    env = MiniGridWrapper(env_name, max_steps=100)

    # Create LLM interface
    if use_mock:
        print("Using MockLLMInterface (random actions)")
        llm = MockLLMInterface()
    else:
        print(f"Connecting to Ollama with model: {model_name}")
        llm_config = {
            'temperature': 0.7,
            'max_tokens': 50,
            'timeout': 30,
        }
        llm = OllamaInterface(model_name, llm_config)

        # Check if Ollama is available
        if not llm.is_available():
            print("\nWarning: Cannot connect to Ollama server!")
            print("Make sure Ollama is running: ollama serve")
            print("Falling back to MockLLMInterface...")
            llm = MockLLMInterface()

    # Create agent
    print(f"\nCreating BaseAgent...")
    agent = BaseAgent(llm, env)

    # Evaluate agent
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    results = quick_evaluate(agent, env, num_episodes=num_episodes, verbose=True)

    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    agent_stats = agent.get_statistics()
    print(f"Agent Type: {agent_stats['agent_type']}")
    print(f"Total Episodes: {agent_stats['total_episodes']}")
    print(f"Success Rate: {agent_stats['success_rate']:.2%}")
    print(f"Average Reward: {agent_stats['avg_reward_per_episode']:.2f}")
    print(f"Average Episode Length: {agent_stats['avg_steps_per_episode']:.1f}")
    print("=" * 80)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
