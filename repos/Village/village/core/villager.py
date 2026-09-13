"""Individual villager implementation - agents with specialized roles."""

import asyncio
from typing import Any, Dict, Optional

from village.exceptions import VillagerError


class Villager:
    """Individual agent with specialized role and capabilities.
    
    A villager represents a single agent that can interact with LLMs,
    maintain memory, and collaborate with other villagers.
    """
    
    def __init__(
        self, 
        name: str, 
        llm_provider: Optional[Any] = None,
        role: str = "general",
        system_prompt: Optional[str] = None
    ) -> None:
        """Initialize a new villager.
        
        Args:
            name: Unique name for the villager
            llm_provider: LLM provider instance for text generation
            role: The role/specialization of the villager
            system_prompt: Custom system prompt for the villager
        """
        self.name = name
        self.llm_provider = llm_provider
        self.role = role
        self.village: Optional[Any] = None
        self._memory: Dict[str, Any] = {}
        self._conversation_history: list = []
        
        # Set default system prompt based on role
        if system_prompt is None:
            self.system_prompt = self._get_default_system_prompt()
        else:
            self.system_prompt = system_prompt
            
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on villager role.
        
        Returns:
            Appropriate system prompt for the role
        """
        role_prompts = {
            "analyst": "You are a data analyst. Analyze information objectively and provide insights based on evidence.",
            "researcher": "You are a researcher. Gather information, verify facts, and provide comprehensive research findings.",
            "writer": "You are a writer. Create clear, engaging, and well-structured written content.",
            "critic": "You are a critic. Provide constructive feedback and identify potential issues or improvements.",
            "coordinator": "You are a coordinator. Organize tasks, manage workflows, and facilitate collaboration.",
            "general": "You are a helpful assistant. Provide accurate and useful responses to questions and tasks."
        }
        return role_prompts.get(self.role, role_prompts["general"])
        
    async def process_task(self, task: str) -> str:
        """Process a task using the villager's capabilities.
        
        Args:
            task: The task to process
            
        Returns:
            The result of processing the task
            
        Raises:
            VillagerError: If no LLM provider is available
        """
        if not self.llm_provider:
            raise VillagerError(f"Villager '{self.name}' has no LLM provider")
            
        try:
            # Create prompt with system context and task
            prompt = f"{self.system_prompt}\n\nTask: {task}"
            
            # Add conversation history context if available
            if self._conversation_history:
                context = "\n".join([
                    f"Previous: {entry['task']} -> {entry['response'][:100]}..."
                    for entry in self._conversation_history[-3:]  # Last 3 interactions
                ])
                prompt = f"{prompt}\n\nContext from recent interactions:\n{context}"
            
            # Process with LLM provider
            response = await self._call_llm(prompt)
            
            # Store in conversation history
            self._conversation_history.append({
                "task": task,
                "response": response,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return response
            
        except Exception as e:
            raise VillagerError(f"Error processing task: {str(e)}")
            
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM provider with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response
        """
        # This is a placeholder - actual implementation depends on LLM provider
        if hasattr(self.llm_provider, 'generate') and self.llm_provider.generate is not None:
            return await self.llm_provider.generate(prompt)
        elif hasattr(self.llm_provider, 'chat') and self.llm_provider.chat is not None:
            return await self.llm_provider.chat(prompt)
        else:
            # Fallback for providers without async methods
            return str(self.llm_provider)
            
    def set_memory(self, key: str, value: Any) -> None:
        """Store a value in the villager's memory.
        
        Args:
            key: The memory key
            value: The value to store
        """
        self._memory[key] = value
        
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the villager's memory.
        
        Args:
            key: The memory key
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        return self._memory.get(key, default)
        
    def clear_memory(self) -> None:
        """Clear all memory."""
        self._memory.clear()
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        
    def get_conversation_history(self) -> list:
        """Get the conversation history.
        
        Returns:
            List of conversation entries
        """
        return self._conversation_history.copy()
        
    def communicate_with(self, other_villager: 'Villager', message: str) -> None:
        """Send a message to another villager.
        
        Args:
            other_villager: The target villager
            message: The message to send
        """
        # Store the communication in both villagers' memory
        self.set_memory(f"sent_to_{other_villager.name}", message)
        other_villager.set_memory(f"received_from_{self.name}", message)
        
    def __repr__(self) -> str:
        """Return string representation of the villager."""
        return f"Villager(name='{self.name}', role='{self.role}')"