"""
Example: Compare LLM agents with RL baselines.

Compares LLM agents (BaseAgent, MemoryAgent, ReflectionAgent) with
traditional RL baselines (PPO, DQN, Random).
"""

import sys
sys.path.append('..')

from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
from agents.reflection_agent import ReflectionAgent
from models.ollama_interface import OllamaInterface, MockLLMInterface
from environments.minigrid_env import MiniGridWrapper
from evaluation.evaluator import Evaluator
from evaluation.rl_baselines import RandomAgent, RLBaselineAgent, SB3_AVAILABLE


def main():
    """Compare LLM agents with RL baselines."""
    print("=" * 80)
    print("LLM Agents vs RL Baselines - LLM-RL Framework")
    print("=" * 80)

    # Configuration
    env_name = "MiniGrid-Empty-8x8-v0"
    model_name = "llama3.2"
    num_eval_episodes = 10
    train_timesteps = 50000  # For RL agents
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

    # Create agents list
    agents = []
    agent_names = []

    # LLM Agents
    print("\nCreating LLM agents...")

    # BaseAgent
    agents.append(BaseAgent(llm, env))
    agent_names.append('LLM-Base')

    # MemoryAgent
    memory_config = {
        'memory_size': 10,
        'success_threshold': 0.5,
        'num_examples': 2,
    }
    agents.append(MemoryAgent(llm, env, memory_config))
    agent_names.append('LLM-Memory')

    # ReflectionAgent
    reflection_config = {
        'memory_size': 10,
        'success_threshold': 0.5,
        'num_examples': 2,
        'reflection_frequency': 5,
        'max_reflections': 3,
    }
    agents.append(ReflectionAgent(llm, env, reflection_config))
    agent_names.append('LLM-Reflection')

    # RL Baselines
    print("\nCreating RL baseline agents...")

    # Random agent
    agents.append(RandomAgent(env))
    agent_names.append('Random')

    # Stable Baselines3 agents (if available)
    if SB3_AVAILABLE:
        print(f"\nTraining RL agents ({train_timesteps} timesteps each)...")

        # PPO
        ppo_agent = RLBaselineAgent(env, algorithm='PPO')
        ppo_agent.train(total_timesteps=train_timesteps, verbose=True)
        agents.append(ppo_agent)
        agent_names.append('PPO')

        # DQN
        dqn_agent = RLBaselineAgent(env, algorithm='DQN')
        dqn_agent.train(total_timesteps=train_timesteps, verbose=True)
        agents.append(dqn_agent)
        agent_names.append('DQN')

        # A2C
        a2c_agent = RLBaselineAgent(env, algorithm='A2C')
        a2c_agent.train(total_timesteps=train_timesteps, verbose=True)
        agents.append(a2c_agent)
        agent_names.append('A2C')
    else:
        print("\nWarning: Stable Baselines3 not available.")
        print("Install with: pip install stable-baselines3")
        print("Only comparing LLM agents with Random baseline.")

    # Create evaluator
    evaluator = Evaluator(env, num_episodes=num_eval_episodes, verbose=True)

    # Evaluate all agents
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)

    results = evaluator.compare_agents(agents, agent_names)

    # Print comparison
    evaluator.print_comparison(results)

    # Print analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find best LLM agent
    llm_agents = {name: res for name, res in results.items() if name.startswith('LLM')}
    if llm_agents:
        best_llm = max(llm_agents.items(), key=lambda x: x[1]['success_rate'])
        print(f"\nBest LLM Agent: {best_llm[0]}")
        print(f"  Success Rate: {best_llm[1]['success_rate']:.2%}")
        print(f"  Mean Reward: {best_llm[1]['mean_reward']:.2f}")

    # Find best RL agent (if available)
    rl_agents = {name: res for name, res in results.items()
                 if name in ['PPO', 'DQN', 'A2C', 'Random']}
    if rl_agents:
        best_rl = max(rl_agents.items(), key=lambda x: x[1]['success_rate'])
        print(f"\nBest RL Baseline: {best_rl[0]}")
        print(f"  Success Rate: {best_rl[1]['success_rate']:.2%}")
        print(f"  Mean Reward: {best_rl[1]['mean_reward']:.2f}")

    # Compare best of each category
    if llm_agents and rl_agents:
        print("\n" + "-" * 80)
        print("Best LLM vs Best RL:")
        if best_llm[1]['success_rate'] > best_rl[1]['success_rate']:
            diff = best_llm[1]['success_rate'] - best_rl[1]['success_rate']
            print(f"  LLM agents outperform RL baselines by {diff:.2%}")
        elif best_rl[1]['success_rate'] > best_llm[1]['success_rate']:
            diff = best_rl[1]['success_rate'] - best_llm[1]['success_rate']
            print(f"  RL baselines outperform LLM agents by {diff:.2%}")
        else:
            print(f"  Performance is tied!")

    # Save results
    output_path = "results/llm_vs_rl_comparison.json"
    evaluator.save_results(results, output_path)

    # Close environment
    env.close()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
