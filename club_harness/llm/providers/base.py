"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to work with the Village framework.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LLM provider.
        
        Args:
            **kwargs: Provider-specific configuration
        """
        self.config = kwargs
        
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """Generate text based on the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass
        
    @abstractmethod
    async def chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """Chat with the model using a conversation format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated response text
        """
        pass
        
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        pass
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: roughly 4 characters per token
        return len(text) // 4
        
    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"{self.__class__.__name__}()"