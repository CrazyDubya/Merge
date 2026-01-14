# Multi-Dimensional Agent Consolidation Plan

## Executive Summary

This plan outlines the strategy for combining 7 agent frameworks into a unified "Club Harness" - an optimized, production-ready agent orchestration system.

## Repository Analysis Summary

| Repository | Core Strength | Key Innovation | Lines of Code |
|------------|--------------|----------------|---------------|
| **TinyTroupe** | Persona simulation | Transactional state + memory consolidation | ~18,400 |
| **llm-council** | Multi-LLM consensus | Anonymous peer review + chairman synthesis | ~3,500 |
| **LisaSimpson** | Deliberative planning | GOAP A* planning + confidence provenance | ~3,400 |
| **Agentic-Hub** | Agent orchestration | Universal command protocol + sandbox isolation | ~11,600 |
| **12-factor-agents** | Production principles | Context window management + stateless reducers | Documentation |
| **hivey** | Swarm intelligence | Self-organizing agents + multi-LLM cost routing | ~4,500 |
| **qwen-code** | Coding agent CLI | Tool system + MCP integration + streaming | ~50,000+ |

---

## Dimension 1: Core Agent Architecture

### Extract & Combine

```
FROM TinyTroupe:
├── Memory System (Episodic + Semantic consolidation)
├── JsonSerializableRegistry (introspectable serialization)
├── ConfigManager (decorator-based config defaults)
└── Transactional decorator (checkpoint/recovery)

FROM LisaSimpson:
├── Confidence provenance tracking
├── GOAP-style A* planner
├── Verification framework (multi-check)
└── Lesson extraction & memory

FROM Agentic-Hub:
├── Universal Agent Interface
├── Command Protocol (works with ANY LLM)
├── Sandbox Manager (isolation levels)
└── Skill System (plugin architecture)

FROM qwen-code:
├── Tool invocation pattern
├── Turn-based agent loop
├── Streaming architecture
└── Subagent system
```

### Unified Agent Model

```python
@dataclass
class UnifiedAgent:
    # Identity (TinyTroupe-inspired)
    name: str
    persona: Dict[str, Any]
    instructions: str

    # Capabilities (Agentic-Hub-inspired)
    skills: List[Skill]
    tools: ToolRegistry

    # State (LisaSimpson-inspired)
    world_state: WorldState
    confidence: ConfidenceTracker
    goals: List[Goal]

    # Memory (TinyTroupe + LisaSimpson)
    episodic_memory: EpisodicMemory
    semantic_memory: SemanticMemory
    lessons: LessonMemory

    # Execution (qwen-code-inspired)
    llm_backend: ContentGenerator
    execution_context: ExecutionContext
```

---

## Dimension 2: Multi-Agent Coordination

### Council/Consensus Layer (from llm-council)

```
Strategies to implement:
├── SimpleRanking - Anonymous peer review + chairman
├── MultiRound - Iterative deliberation with revision
├── WeightedVoting - Performance-based influence
└── ReasoningAware - Dual evaluation for reasoning models
```

### Swarm Coordination (from hivey)

```
Hierarchical tiers:
├── Meta-Agents (Organizer, Judge, Inspirator)
├── Supervisor Agents (domain coordinators)
└── Worker Agents (task executors)

Self-organization:
├── Dynamic agent proposal
├── Experience-based learning
└── Cost-aware LLM routing
```

### Task Decomposition (combined)

```
Agentic-Hub workflow decomposition
    + hivey hierarchical assignment
    + LisaSimpson GOAP planning
    + 12-factor micro-agent pattern
    = Intelligent Task Graph
```

---

## Dimension 3: LLM Integration Strategy

### Multi-Provider Architecture

```python
class LLMRouter:
    """
    Combines:
    - qwen-code's ContentGenerator abstraction
    - hivey's cost-aware tier routing
    - llm-council's multi-model orchestration
    """

    providers = {
        "openrouter": OpenRouterBackend,  # 400+ models
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "ollama": OllamaBackend,  # Local models
        "qwen": QwenBackend,
    }

    tiers = {
        "flash": ["gemini-flash", "gpt-4o-mini", "haiku"],
        "standard": ["gpt-4o", "sonnet", "qwen-plus"],
        "reasoning": ["o1", "o3", "deepseek-r1"],
        "advanced": ["opus", "gpt-4.5"],
    }

    def route(self, task_complexity: str, cost_priority: bool):
        # Intelligent routing based on task + cost
        pass
```

### OpenRouter Integration (Primary)

```python
# Environment: OPENROUTER_API_KEY available
# Free/cheap models for testing:

FREE_MODELS = [
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

CHEAP_MODELS = [
    "google/gemini-flash-1.5",  # $0.075/M input
    "anthropic/claude-3-haiku", # $0.25/M input
    "openai/gpt-4o-mini",       # $0.15/M input
]
```

---

## Dimension 4: Tool & Execution System

### Unified Tool Protocol

```python
# Combines qwen-code tool pattern + Agentic-Hub command protocol

class UnifiedTool:
    name: str
    description: str
    parameters: JSONSchema

    # From qwen-code
    def should_confirm(self) -> ToolConfirmation | False
    def execute(self, signal: AbortSignal) -> ToolResult

    # From Agentic-Hub
    def to_command(self) -> Command  # Works with non-tool LLMs
    def from_natural_language(self, text: str) -> ToolCall  # Parse NL
```

### Execution Modes

```
1. Tool-Native (OpenAI, Anthropic, Qwen)
   - Direct function calling
   - Structured JSON output

2. Command Protocol (Any LLM)
   - Text block parsing
   - Natural language fallback

3. Sandbox Execution (Agentic-Hub)
   - Process isolation
   - Container isolation
   - Resource limits
```

---

## Dimension 5: State & Memory Management

### Layered Memory Architecture

```
┌─────────────────────────────────────────┐
│ Working Memory (qwen-code)              │
│ - Current task context                  │
│ - Recent tool outputs                   │
│ - Short-term findings                   │
├─────────────────────────────────────────┤
│ Episodic Memory (TinyTroupe)            │
│ - Event sequences                       │
│ - Automatic consolidation               │
│ - Fixed prefix + lookback window        │
├─────────────────────────────────────────┤
│ Semantic Memory (TinyTroupe + hivey)    │
│ - Vector-indexed knowledge              │
│ - Similarity search                     │
│ - Experience retrieval                  │
├─────────────────────────────────────────┤
│ Lesson Memory (LisaSimpson)             │
│ - Extracted insights                    │
│ - Success/failure patterns              │
│ - Confidence with decay                 │
└─────────────────────────────────────────┘
```

### State Management (12-factor principles)

```python
# Stateless reducer pattern
class AgentStep:
    def __call__(self, state: AgentState) -> AgentState:
        # Pure function: input → output
        # Enables: pause, resume, replay, scale
        pass

# Event-sourced state
class EventLog:
    events: List[Event]  # Append-only

    def reconstruct_state(self, timestamp: datetime) -> AgentState:
        # Replay to any point
        pass
```

---

## Dimension 6: Quality & Verification

### Multi-Level Verification (LisaSimpson)

```python
class VerificationPlan:
    checks: List[Check]
    # TypeCheck - Static analysis
    # TestCheck - Test suites
    # LintCheck - Code quality
    # SemanticCheck - LLM-based property verification

    confidence_threshold: float
```

### Self-Evaluation (Agentic-Hub + hivey)

```python
class SelfEvaluator:
    dimensions = [
        "task_success",
        "output_quality",
        "efficiency",
        "safety",
        "tool_usage",
        "collaboration",
    ]

    def evaluate(self, execution: Execution) -> EvaluationResult:
        # JudgeAgent from hivey
        # Proposition framework from TinyTroupe
        pass
```

### Quality Gates (12-factor)

```
Before execution:
├── Confidence check (LisaSimpson)
├── Plan quality evaluation
└── Human approval for high-risk

During execution:
├── Loop detection
├── Error compaction
└── Context utilization monitoring (<40%)

After execution:
├── Verification checks
├── Lesson extraction
└── Performance metrics update
```

---

## Dimension 7: Production Readiness

### 12-Factor Compliance

| Factor | Implementation |
|--------|---------------|
| 1. NL→Tool Calls | Universal command protocol |
| 2. Own Prompts | Version-controlled templates |
| 3. Own Context | Active context management |
| 4. Tools=Outputs | Structured JSON routing |
| 5. Unified State | Event-sourced state model |
| 6. Pause/Resume | Checkpoint/recovery |
| 7. Human Contact | Tool-based escalation |
| 8. Own Control Flow | Explicit routing logic |
| 9. Compact Errors | Intelligent error summarization |
| 10. Small Agents | Micro-agent composition |
| 11. Trigger Anywhere | Event-driven activation |
| 12. Stateless Reducer | Pure function architecture |

---

## Implementation Phases

### Phase 1: Foundation (Club Harness Core)
- [ ] Unified configuration system
- [ ] LLM router with OpenRouter
- [ ] Basic tool system
- [ ] Simple agent loop

### Phase 2: Memory & State
- [ ] Episodic memory implementation
- [ ] State management with events
- [ ] Confidence tracking
- [ ] Lesson extraction

### Phase 3: Multi-Agent
- [ ] Council consensus strategies
- [ ] Swarm coordination
- [ ] Task decomposition
- [ ] Inter-agent communication

### Phase 4: Production Features
- [ ] Sandbox isolation
- [ ] Self-evaluation
- [ ] Verification framework
- [ ] Human-in-the-loop

### Phase 5: Optimization
- [ ] Cost-aware routing
- [ ] Context compression
- [ ] Performance tuning
- [ ] Caching strategies

---

## Club Harness Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     CLUB HARNESS                             │
├──────────────────────────────────────────────────────────────┤
│  CLI Interface (qwen-code inspired)                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Interactive REPL │ Non-Interactive │ API Server        │ │
│  └─────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                         │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Council      │ Swarm        │ Task Planner             │ │
│  │ (consensus)  │ (hierarchy)  │ (GOAP)                   │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Unified      │ Tool         │ Memory                   │ │
│  │ Agent        │ Registry     │ System                   │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  LLM Layer                                                   │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ OpenRouter   │ Ollama       │ Direct APIs              │ │
│  │ (400+ models)│ (local)      │ (OpenAI, Anthropic)     │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  Execution Layer                                             │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Sandbox      │ Verification │ Event Log                │ │
│  │ Manager      │ Framework    │ (audit trail)            │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Build Club Harness Core** - Start with minimal viable agent
2. **Test with OpenRouter** - Validate LLM integration
3. **Add Memory System** - Implement episodic + semantic memory
4. **Add Council/Swarm** - Multi-agent coordination
5. **Production Hardening** - Apply 12-factor principles

---

## File Structure

```
/club_harness/
├── __init__.py
├── core/
│   ├── agent.py          # UnifiedAgent class
│   ├── config.py         # Configuration management
│   ├── state.py          # State management
│   └── types.py          # Core type definitions
├── llm/
│   ├── router.py         # LLM routing logic
│   ├── openrouter.py     # OpenRouter backend
│   └── backends/         # Provider backends
├── tools/
│   ├── registry.py       # Tool registry
│   ├── protocol.py       # Command protocol
│   └── builtin/          # Built-in tools
├── memory/
│   ├── episodic.py       # Episodic memory
│   ├── semantic.py       # Semantic memory
│   └── lessons.py        # Lesson memory
├── orchestration/
│   ├── council.py        # Council consensus
│   ├── swarm.py          # Swarm coordination
│   └── planner.py        # GOAP planner
├── verification/
│   ├── checks.py         # Verification checks
│   └── evaluator.py      # Self-evaluation
└── cli/
    ├── main.py           # CLI entry point
    └── ui.py             # Terminal UI
```
