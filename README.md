# LLM-RL Agent Framework

A framework for exploring **out-of-the-box Large Language Models (LLMs)** as reinforcement learning (RL) policy generators on MiniGrid navigation tasks.

## Overview

This project investigates whether pre-trained LLMs like Llama3.2 (via Ollama) can effectively solve RL tasks through **structured prompting**, **episodic memory**, and **self-reflection** - without any fine-tuning or gradient-based learning.

### Motivation

Traditional RL algorithms (PPO, DQN, etc.) require extensive training to learn policies. This framework explores an alternative approach:
- **Zero-shot learning**: LLMs solve tasks using their pre-trained knowledge
- **In-context learning**: Agents improve through memory and reflection
- **Natural language reasoning**: Decision-making is interpretable and explainable

### Key Features

- **Three agent architectures** with increasing sophistication
- **Local LLM inference** via Ollama (Llama3.2)
- **MiniGrid environment** wrapper for LLM-friendly observations
- **Comprehensive evaluation** framework with RL baselines
- **Modular design** for easy extension and experimentation

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) for local LLM inference

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/LLM-RL-framework.git
cd LLM-RL-framework

# Install Python packages
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Step 2: Install and Setup Ollama

```bash
# Download and install Ollama from https://ollama.ai/

# Pull Llama3.2 model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

## Quick Start

### Run BaseAgent

Simple example using BaseAgent on an empty grid:

```bash
cd examples
python run_base_agent.py
```

### Compare All Agent Types

Compare BaseAgent, MemoryAgent, and ReflectionAgent:

```bash
python run_all_agents.py
```

### Compare with RL Baselines

Compare LLM agents with PPO, DQN, and Random baselines:

```bash
python compare_with_baselines.py
```

## Architecture

### Agent Types

#### 1. BaseAgent

Uses **structured prompting** to convert observations to actions:
- Formats MiniGrid grids in human-readable text
- Provides action choices to LLM
- Parses LLM response to extract action
- **No learning** between episodes

```
Observation → LLM Prompt → LLM Response → Action
```

#### 2. MemoryAgent

Extends BaseAgent with **episodic memory**:
- Stores past episodes with success/failure labels
- Augments prompts with successful episode examples
- Learns from experience via in-context learning
- Configurable memory size and retrieval

```
Observation + Memory → LLM Prompt → LLM Response → Action
              ↑                                        ↓
              └────────── Episode Storage ←────────────┘
```

#### 3. ReflectionAgent

Extends MemoryAgent with **self-reflection**:
- Periodically analyzes successful vs failed episodes
- Generates high-level insights about strategies
- Augments prompts with both memory AND reflections
- Configurable reflection frequency

```
Observation + Memory + Reflections → LLM Prompt → LLM Response → Action
              ↑          ↑                                         ↓
              └──────────┴───────── Episode Analysis ←─────────────┘
```

## Project Structure

```
LLM-RL-framework/
├── agents/
│   ├── base_agent.py          # BaseAgent implementation
│   ├── memory_agent.py        # MemoryAgent with episodic memory
│   └── reflection_agent.py    # ReflectionAgent with self-reflection
├── models/
│   └── ollama_interface.py    # Ollama API interface & MockLLM
├── environments/
│   └── minigrid_env.py        # MiniGrid wrapper & task suite
├── evaluation/
│   ├── evaluator.py           # Evaluation framework
│   └── rl_baselines.py        # RL baseline agents (PPO, DQN, etc.)
├── configs/
│   └── default_config.yaml    # Default configuration
├── utils/
│   └── config_loader.py       # Configuration loader
├── examples/
│   ├── run_base_agent.py      # Simple BaseAgent example
│   ├── run_all_agents.py      # Compare all agent types
│   └── compare_with_baselines.py  # LLM vs RL comparison
├── tests/
│   └── ...                    # Unit tests
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Usage Examples

### Creating Agents

```python
from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
from agents.reflection_agent import ReflectionAgent
from models.ollama_interface import OllamaInterface
from environments.minigrid_env import MiniGridWrapper

# Create environment
env = MiniGridWrapper("MiniGrid-Empty-8x8-v0", max_steps=100)

# Create LLM interface
llm = OllamaInterface(model_name="llama3.2", config={
    'temperature': 0.7,
    'max_tokens': 50,
})

# Create agents
base_agent = BaseAgent(llm, env)

memory_agent = MemoryAgent(llm, env, agent_config={
    'memory_size': 10,
    'num_examples': 2,
})

reflection_agent = ReflectionAgent(llm, env, agent_config={
    'memory_size': 10,
    'reflection_frequency': 5,
})
```

### Running Evaluation

```python
from evaluation.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(env, num_episodes=10, verbose=True)

# Evaluate single agent
results = evaluator.evaluate_agent(base_agent)

# Compare multiple agents
agents = [base_agent, memory_agent, reflection_agent]
agent_names = ['BaseAgent', 'MemoryAgent', 'ReflectionAgent']
comparison = evaluator.compare_agents(agents, agent_names)

# Print comparison table
evaluator.print_comparison(comparison)

# Save results
evaluator.save_results(comparison, "results/comparison.json")
```

### Using Configuration Files

```python
from utils.config_loader import ConfigLoader

# Load config
config = ConfigLoader("configs/default_config.yaml")

# Access config values
llm_config = config.get_llm_config()
agent_config = config.get_agent_config()
env_name = config.get("environment.name")

# Update config
config.update("agent.type", "reflection")
config.save("configs/my_config.yaml")
```

## MiniGrid Tasks

### Easy Tasks (Simple navigation, no obstacles)
- `MiniGrid-Empty-5x5-v0`
- `MiniGrid-Empty-6x6-v0`
- `MiniGrid-Empty-8x8-v0`

### Medium Tasks (Obstacles, doors, simple interactions)
- `MiniGrid-FourRooms-v0`
- `MiniGrid-DoorKey-5x5-v0`
- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-SimpleCrossingS9N1-v0`

### Hard Tasks (Complex planning, multiple interactions)
- `MiniGrid-DoorKey-8x8-v0`
- `MiniGrid-MultiRoom-N2-S4-v0`
- `MiniGrid-Unlock-v0`
- `MiniGrid-UnlockPickup-v0`

## Evaluation Metrics

The framework tracks the following metrics:

- **Success Rate**: Percentage of episodes where agent reaches goal
- **Mean Reward**: Average cumulative reward per episode
- **Mean Episode Length**: Average number of steps per episode
- **Std Reward/Length**: Standard deviation of reward/length
- **Total Time**: Total evaluation time

## Example Results

Example comparison on `MiniGrid-Empty-8x8-v0` (10 episodes each):

```
================================================================================
AGENT COMPARISON RESULTS
================================================================================
Agent                          Success Rate    Mean Reward     Mean Length
--------------------------------------------------------------------------------
ReflectionAgent                60.0%           0.48 ± 0.51     45.2 ± 12.3
MemoryAgent                    50.0%           0.40 ± 0.52     52.1 ± 15.7
PPO                            40.0%           0.32 ± 0.48     58.3 ± 18.2
BaseAgent                      30.0%           0.24 ± 0.44     64.7 ± 20.1
DQN                            20.0%           0.16 ± 0.37     72.5 ± 22.8
Random                         10.0%           0.08 ± 0.28     89.2 ± 8.5
================================================================================
```

*Note: Results are illustrative. Actual performance depends on LLM model, prompts, and task complexity.*

## Configuration

The framework uses YAML configuration files for easy experimentation.

### Default Configuration

See `configs/default_config.yaml` for all options:

```yaml
llm:
  model_name: "llama3.2"
  temperature: 0.7
  max_tokens: 100

agent:
  type: "reflection"
  memory_size: 10
  reflection_frequency: 5

environment:
  name: "MiniGrid-Empty-8x8-v0"
  max_steps: 100

evaluation:
  num_episodes: 10
  save_results: true
```

### Custom Configuration

Create custom configs for different experiments:

```python
from utils.config_loader import ConfigLoader

config = ConfigLoader()
config.update("llm.model_name", "llama3.1")
config.update("agent.type", "memory")
config.update("environment.name", "MiniGrid-DoorKey-8x8-v0")
config.save("configs/experiment1.yaml")
```

## Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=. tests/
```

## Limitations

- **Inference Speed**: LLM inference is slower than RL forward passes
- **Context Length**: Limited episode history due to context windows
- **Generalization**: Performance depends heavily on prompt engineering
- **Determinism**: LLM outputs may vary between runs
- **Scalability**: May not scale to complex, high-dimensional tasks

## Future Work

- [ ] Support for additional LLM backends (OpenAI, Anthropic, local models)
- [ ] Advanced prompting techniques (chain-of-thought, tree-of-thoughts)
- [ ] Multi-modal observations (visual understanding)
- [ ] Fine-tuning experiments for RL tasks
- [ ] Hybrid LLM-RL approaches
- [ ] Extended task suite (Atari, MuJoCo, custom environments)
- [ ] Parallel episode evaluation for faster benchmarking
- [ ] Interactive visualization tools

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_rl_framework,
  title = {LLM-RL Framework: Exploring LLMs as RL Policy Generators},
  author = {LLM-RL Framework Contributors},
  year = {2024},
  url = {https://github.com/yourusername/LLM-RL-framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid) for the environment suite
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL baselines
- The open-source AI/ML community

## Contact

For questions, issues, or discussions:
- GitHub Issues: [https://github.com/yourusername/LLM-RL-framework/issues](https://github.com/yourusername/LLM-RL-framework/issues)
- Email: your.email@example.com

---

Built with curiosity about the intersection of LLMs and reinforcement learning.
