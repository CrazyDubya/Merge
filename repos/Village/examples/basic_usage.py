"""Basic usage example for the Village framework."""

import asyncio
from village import Village, Villager


class MockLLMProvider:
    """Mock LLM provider for demonstration purposes."""
    
    def __init__(self, name: str = "Mock"):
        self.name = name
        
    async def generate(self, prompt: str) -> str:
        """Generate a mock response."""
        if "analyze" in prompt.lower():
            return f"[{self.name}] Analysis complete: The data shows positive trends."
        elif "research" in prompt.lower():
            return f"[{self.name}] Research findings: Multiple sources confirm the hypothesis."
        else:
            return f"[{self.name}] Task completed successfully."


async def main():
    """Demonstrate basic Village usage."""
    print("🏘️ Welcome to Village - LLM Interaction Framework")
    print("=" * 50)
    
    # Create a village
    village = Village("AI Research Village")
    print(f"Created village: {village}")
    
    # Create LLM providers
    provider1 = MockLLMProvider("GPT-Mock")
    provider2 = MockLLMProvider("Claude-Mock")
    
    # Create villagers with different roles
    analyst = Villager("DataAnalyst", provider1, role="analyst")
    researcher = Villager("Researcher", provider2, role="researcher")
    coordinator = Villager("Coordinator", provider1, role="coordinator")
    
    # Add villagers to village
    village.add_villager(analyst)
    village.add_villager(researcher)
    village.add_villager(coordinator)
    
    print(f"\nVillagers in {village.name}:")
    for name in village.list_villagers():
        villager = village.get_villager(name)
        print(f"  - {name} ({villager.role})")
    
    # Demonstrate individual task processing
    print("\n📋 Individual Task Processing:")
    print("-" * 30)
    
    task1 = "Analyze recent market trends in AI technology"
    result1 = await analyst.process_task(task1)
    print(f"Analyst: {result1}")
    
    task2 = "Research the impact of LLMs on productivity"
    result2 = await researcher.process_task(task2)
    print(f"Researcher: {result2}")
    
    # Demonstrate collaborative work
    print("\n🤝 Collaborative Task:")
    print("-" * 20)
    
    collaborative_task = "Create a comprehensive report on AI adoption in enterprises"
    result = await village.collaborate(collaborative_task)
    print("Collaborative Result:")
    print(result)
    
    # Demonstrate memory and communication
    print("\n🧠 Memory and Communication:")
    print("-" * 30)
    
    # Store information in memory
    analyst.set_memory("last_analysis", "Positive growth trend identified")
    researcher.set_memory("key_finding", "70% productivity increase reported")
    
    # Villagers communicate
    analyst.communicate_with(researcher, "I found positive trends in my analysis")
    researcher.communicate_with(coordinator, "Analysis shows 70% productivity boost")
    
    print(f"Analyst memory: {analyst.get_memory('last_analysis')}")
    print(f"Researcher received: {researcher.get_memory('received_from_DataAnalyst')}")
    print(f"Coordinator received: {coordinator.get_memory('received_from_Researcher')}")
    
    # Show conversation history
    print("\n📚 Conversation History:")
    print("-" * 25)
    
    history = analyst.get_conversation_history()
    for i, entry in enumerate(history, 1):
        print(f"{i}. Task: {entry['task'][:50]}...")
        print(f"   Response: {entry['response'][:50]}...")
    
    # Show task history
    print("\n📊 Village Task History:")
    print("-" * 25)
    
    task_history = village.get_task_history()
    for i, task_record in enumerate(task_history, 1):
        print(f"{i}. {task_record['task'][:50]}...")
        print(f"   Villagers: {', '.join(task_record['villagers'])}")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())