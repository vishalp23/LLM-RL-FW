"""
Example: Compare all three agent types.

Demonstrates differences between BaseAgent, MemoryAgent, and ReflectionAgent.
"""

import sys
sys.path.append('..')

from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
from agents.reflection_agent import ReflectionAgent
from models.ollama_interface import OllamaInterface, MockLLMInterface
from environments.minigrid_env import MiniGridWrapper
from evaluation.evaluator import Evaluator


def main():
    """Compare all agent types."""
    print("=" * 80)
    print("Multi-Agent Comparison - LLM-RL Framework")
    print("=" * 80)

    # Configuration
    env_name = "MiniGrid-Empty-8x8-v0"
    model_name = "llama3.2"
    num_episodes = 10
    use_mock = False  # Set to True to use mock LLM

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

        if not llm.is_available():
            print("\nWarning: Cannot connect to Ollama server!")
            print("Make sure Ollama is running: ollama serve")
            print("Falling back to MockLLMInterface...")
            llm = MockLLMInterface()

    # Create agents
    print("\nCreating agents...")

    # BaseAgent - no memory or reflection
    base_agent = BaseAgent(llm, env)

    # MemoryAgent - with episodic memory
    memory_config = {
        'memory_size': 10,
        'success_threshold': 0.5,
        'num_examples': 2,
    }
    memory_agent = MemoryAgent(llm, env, memory_config)

    # ReflectionAgent - with memory and self-reflection
    reflection_config = {
        'memory_size': 10,
        'success_threshold': 0.5,
        'num_examples': 2,
        'reflection_frequency': 5,
        'max_reflections': 3,
    }
    reflection_agent = ReflectionAgent(llm, env, reflection_config)

    # Create evaluator
    evaluator = Evaluator(env, num_episodes=num_episodes, verbose=True)

    # Evaluate all agents
    agents = [base_agent, memory_agent, reflection_agent]
    agent_names = ['BaseAgent', 'MemoryAgent', 'ReflectionAgent']

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)

    results = evaluator.compare_agents(agents, agent_names)

    # Print comparison
    evaluator.print_comparison(results)

    # Print detailed statistics for each agent
    print("\n" + "=" * 80)
    print("DETAILED AGENT STATISTICS")
    print("=" * 80)

    for agent, name in zip(agents, agent_names):
        print(f"\n{name}:")
        print("-" * 40)
        stats = agent.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    # Save results
    output_path = "results/agent_comparison.json"
    evaluator.save_results(results, output_path)

    # Close environment
    env.close()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
