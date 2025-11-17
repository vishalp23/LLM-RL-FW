"""
Models package for LLM-RL Framework.

Provides LLM interface implementations.
"""

from models.ollama_interface import OllamaInterface, MockLLMInterface

__all__ = [
    'OllamaInterface',
    'MockLLMInterface',
]
