"""
Ollama Interface for LLM-RL Framework.

Provides interface to Ollama API for local LLM inference.
"""

import time
import requests
from typing import Dict, List, Any, Optional
import json


class OllamaInterface:
    """
    Interface to Ollama API for LLM inference.

    Connects to local Ollama server and provides methods for:
    - Text generation
    - Chat-based interactions
    - Configurable parameters (temperature, max_tokens, etc.)
    - Retry logic for robustness
    """

    def __init__(self, model_name: str = "llama3.2", config: Dict[str, Any] = None):
        """
        Initialize Ollama interface.

        Args:
            model_name: Name of Ollama model to use (default: "llama3.2")
            config: Configuration dictionary
                - host: Ollama server host (default: "http://localhost:11434")
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 100)
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Maximum retry attempts (default: 3)
                - retry_delay: Delay between retries in seconds (default: 1)
        """
        self.model_name = model_name
        self.config = config or {}

        # API configuration
        self.host = self.config.get('host', 'http://localhost:11434')
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 100)
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1)

        # API endpoints
        self.generate_endpoint = f"{self.host}/api/generate"
        self.chat_endpoint = f"{self.host}/api/chat"

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: Input prompt string
            **kwargs: Additional parameters to override defaults
                (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            Exception: If generation fails after all retries
        """
        # Prepare request payload
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                return result.get('response', '').strip()

            except requests.exceptions.Timeout:
                print(f"Warning: Request timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Generation failed: Request timeout")

            except requests.exceptions.ConnectionError:
                print(f"Warning: Connection error on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Generation failed: Cannot connect to Ollama server. "
                                    "Make sure Ollama is running (ollama serve)")

            except requests.exceptions.RequestException as e:
                print(f"Warning: Request failed on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"Generation failed: {e}")

            except Exception as e:
                raise Exception(f"Unexpected error during generation: {e}")

        raise Exception("Generation failed after all retries")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat completion from message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                Example: [{'role': 'user', 'content': 'Hello!'}]
            **kwargs: Additional parameters to override defaults

        Returns:
            Generated response text

        Raises:
            Exception: If chat generation fails after all retries
        """
        # Prepare request payload
        payload = {
            'model': self.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', self.temperature),
                'num_predict': kwargs.get('max_tokens', self.max_tokens),
            }
        }

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                message = result.get('message', {})
                return message.get('content', '').strip()

            except requests.exceptions.Timeout:
                print(f"Warning: Request timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Chat failed: Request timeout")

            except requests.exceptions.ConnectionError:
                print(f"Warning: Connection error on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception("Chat failed: Cannot connect to Ollama server. "
                                    "Make sure Ollama is running (ollama serve)")

            except requests.exceptions.RequestException as e:
                print(f"Warning: Request failed on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"Chat failed: {e}")

            except Exception as e:
                raise Exception(f"Unexpected error during chat: {e}")

        raise Exception("Chat failed after all retries")

    def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names

        Raises:
            Exception: If cannot connect to server
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            result = response.json()
            models = result.get('models', [])
            return [model.get('name', '') for model in models]
        except Exception as e:
            raise Exception(f"Failed to list models: {e}")

    def __str__(self) -> str:
        """String representation."""
        return f"OllamaInterface(model={self.model_name}, host={self.host})"


class MockLLMInterface:
    """
    Mock LLM interface for testing without Ollama.

    Provides same interface as OllamaInterface but returns
    simple mock responses for testing purposes.
    """

    def __init__(self, model_name: str = "mock", config: Dict[str, Any] = None):
        """
        Initialize mock interface.

        Args:
            model_name: Model name (ignored, for compatibility)
            config: Configuration (ignored, for compatibility)
        """
        self.model_name = model_name
        self.config = config or {}
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock response.

        Args:
            prompt: Input prompt
            **kwargs: Ignored

        Returns:
            Mock response (random action 0-6)
        """
        import random
        self.call_count += 1

        # Return random action for testing
        action = random.randint(0, 6)
        return str(action)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate mock chat response.

        Args:
            messages: Message history
            **kwargs: Ignored

        Returns:
            Mock response
        """
        return self.generate("mock_prompt")

    def is_available(self) -> bool:
        """Always available."""
        return True

    def list_models(self) -> List[str]:
        """Return mock model list."""
        return ["mock"]

    def __str__(self) -> str:
        """String representation."""
        return f"MockLLMInterface(calls={self.call_count})"
