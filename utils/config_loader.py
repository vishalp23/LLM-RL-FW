"""
Configuration Loader for LLM-RL Framework.

Provides utilities for loading and parsing YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Configuration loader and manager.

    Loads YAML configuration files and provides easy access to config values.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file (default: configs/default_config.yaml)
        """
        if config_path is None:
            # Use default config
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "default_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys with dot notation (e.g., "llm.temperature").

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration section.

        Returns:
            LLM configuration dictionary
        """
        return self.config.get('llm', {})

    def get_agent_config(self) -> Dict[str, Any]:
        """
        Get agent configuration section.

        Returns:
            Agent configuration dictionary
        """
        return self.config.get('agent', {})

    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment configuration section.

        Returns:
            Environment configuration dictionary
        """
        return self.config.get('environment', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration section.

        Returns:
            Evaluation configuration dictionary
        """
        return self.config.get('evaluation', {})

    def get_rl_baselines_config(self) -> Dict[str, Any]:
        """
        Get RL baselines configuration section.

        Returns:
            RL baselines configuration dictionary
        """
        return self.config.get('rl_baselines', {})

    def update(self, key: str, value: Any):
        """
        Update configuration value.

        Supports nested keys with dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config

        # Navigate to nested dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

    def save(self, output_path: str = None):
        """
        Save configuration to YAML file.

        Args:
            output_path: Path to save config (default: self.config_path)
        """
        if output_path is None:
            output_path = self.config_path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def __str__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path={self.config_path})"

    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()


def load_config(config_path: str = None) -> ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to YAML config file (default: configs/default_config.yaml)

    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


def create_default_config(output_path: str):
    """
    Create a default configuration file.

    Args:
        output_path: Path to save default config
    """
    default_config = {
        'llm': {
            'model_name': 'llama3.2',
            'host': 'http://localhost:11434',
            'temperature': 0.7,
            'max_tokens': 100,
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1,
        },
        'agent': {
            'type': 'reflection',
            'memory_size': 10,
            'success_threshold': 0.5,
            'num_examples': 2,
            'reflection_frequency': 5,
            'max_reflections': 5,
        },
        'environment': {
            'name': 'MiniGrid-Empty-8x8-v0',
            'max_steps': 100,
            'render_mode': None,
            'difficulty': 'easy',
        },
        'evaluation': {
            'num_episodes': 10,
            'verbose': True,
            'save_results': True,
            'results_dir': 'results',
        },
        'rl_baselines': {
            'algorithms': ['PPO', 'DQN', 'A2C'],
            'train_timesteps': 100000,
            'learning_rate': 0.0003,
            'verbose': 0,
        },
        'logging': {
            'enabled': True,
            'log_dir': 'logs',
            'log_level': 'INFO',
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Default configuration created at: {output_path}")
