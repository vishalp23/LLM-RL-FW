"""
Evaluation package for LLM-RL Framework.

Provides evaluation tools and baseline agents.
"""

from evaluation.evaluator import Evaluator, quick_evaluate, quick_compare
from evaluation.rl_baselines import RandomAgent, RLBaselineAgent, create_baseline_agents

__all__ = [
    'Evaluator',
    'quick_evaluate',
    'quick_compare',
    'RandomAgent',
    'RLBaselineAgent',
    'create_baseline_agents',
]
