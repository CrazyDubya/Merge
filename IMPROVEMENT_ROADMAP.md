# Club Harness Improvement Roadmap

## Current State

**Integrated Components:**
- Core agent architecture with persona system
- OpenRouter backend with 400+ model access
- LLM router with tier-based selection and retry logic
- Memory system (episodic, semantic, lessons, working memory)
- Tool system with calculator, web search, shell tools
- Council consensus with simple ranking strategy
- Loop detection (newly added from TinyTroupe)
- Streaming support (sync and async)

**Test Coverage:**
- Basic integration tests (6/6 passing)
- Advanced feature tests (5/5 passing)
- Comprehensive test suite

---

## Phase 1: Foundation (High Priority)

### 1.1 Advanced Memory Consolidation
**Source:** TinyTroupe (`agent/memory.py` - 49KB)
**Impact:** High - enables long-running agents

Features to integrate:
- [ ] Fixed prefix + lookback window pattern
- [ ] Automatic memory consolidation
- [ ] Reflection consolidation
- [ ] Memory compression for old episodes
- [ ] Configurable consolidation windows

### 1.2 Semantic Caching
**Source:** TinyTroupe (`caching/semantic_cache.py`)
**Impact:** High - reduces API costs, improves performance

Features to integrate:
- [ ] Embedding-based fuzzy cache hits
- [ ] Similarity threshold matching
- [ ] Hybrid exact + semantic lookup
- [ ] Cache statistics and monitoring

### 1.3 GOAP-Style Planning
**Source:** LisaSimpson (`deliberative_agent/planning.py`)
**Impact:** High - enables deliberative reasoning

Features to integrate:
- [ ] A* pathfinding for action sequencing
- [ ] Plan class with verification points
- [ ] Action preconditions and effects
- [ ] Cost estimation for actions
- [ ] Plan verification predicates

---

## Phase 2: Features (Medium Priority)

### 2.1 Advanced Consensus Strategies
**Source:** llm-council (`backend/strategies/`)
**Impact:** Medium - improves answer quality

Strategies to integrate:
- [ ] MultiRoundStrategy (iterative refinement)
- [ ] WeightedVotingStrategy (performance-based)
- [ ] ReasoningAwareStrategy (for o1/o3 models)
- [ ] StrategyRecommender (auto-select)
- [ ] Query classifier for automatic routing

### 2.2 Verification Framework
**Source:** LisaSimpson (`deliberative_agent/verification.py`)
**Impact:** Medium - enables quality checks

Features to integrate:
- [ ] TypeCheck, TestCheck, SemanticCheck classes
- [ ] Async verification support
- [ ] Confidence-based threshold evaluation
- [ ] Verification result aggregation

### 2.3 Sandbox Manager
**Source:** Agentic-Hub (`harness/sandbox/sandbox_manager.py` - 25KB)
**Impact:** High - essential for safe execution

Features to integrate:
- [ ] SandboxType enum (PRIVATE, SHARED, READONLY, EPHEMERAL)
- [ ] IsolationLevel (NONE, PROCESS, CONTAINER, VM)
- [ ] ResourceLimits (CPU, memory, disk, network)
- [ ] SandboxSnapshot for point-in-time recovery
- [ ] File manifest tracking

### 2.4 Knowledge Base System
**Source:** hivey (SQLite-backed)
**Impact:** Medium - persistent learning

Features to integrate:
- [ ] Vector embedding storage
- [ ] Semantic similarity search
- [ ] Experience retrieval
- [ ] Cross-session learning

---

## Phase 3: Production Features

### 3.1 Error Handling Framework
**Source:** hivey (`error_handling.py`)
**Impact:** Medium - production robustness

Features to integrate:
- [ ] Custom exception hierarchy
- [ ] ErrorDetails dataclass
- [ ] safe_execute decorator
- [ ] Input validation system
- [ ] Error recovery strategies

### 3.2 Advanced CLI
**Source:** Agentic-Hub (`harness/cli.py` - 34KB)
**Impact:** Medium - better UX

Features to integrate:
- [ ] Interactive REPL mode
- [ ] Server mode with proper shutdown
- [ ] Rich terminal UI
- [ ] Provider/model listing
- [ ] Command history

### 3.3 Profiling & Monitoring
**Source:** TinyTroupe (`profiling.py`, `monitoring/memory_monitor.py`)
**Impact:** Medium - observability

Features to integrate:
- [ ] MemoryMonitor with alerting
- [ ] MemoryProfiler for detailed analysis
- [ ] Performance telemetry
- [ ] Abnormal growth detection

### 3.4 FastAPI Service
**Source:** hivey (`swarm_service.py`)
**Impact:** Medium - deployment ready

Features to integrate:
- [ ] REST API endpoints
- [ ] API key authentication
- [ ] Async operations
- [ ] Proper logging/error handling
- [ ] Health checks

---

## Phase 4: Advanced Features

### 4.1 Swarm Coordination
**Source:** hivey (`swarms.py`)
**Impact:** Low - advanced use cases

Features to integrate:
- [ ] Hierarchical agent organization
- [ ] Meta-agents (Organizer, Judge, Inspirator)
- [ ] Worker and supervisor tiers
- [ ] Self-organizing behavior

### 4.2 Scenario Templates
**Source:** Agentic-Hub (`scenarios/`)
**Impact:** Low - specialized use cases

Templates to consider:
- [ ] Prison scenario
- [ ] Physics lab scenario
- [ ] Corporate scenario
- [ ] D&D scenario
- [ ] Theatre scenario

### 4.3 Confidence Decay System
**Source:** LisaSimpson
**Impact:** Medium - better uncertainty modeling

Features to integrate:
- [ ] Time-based confidence decay
- [ ] Half-life calculations
- [ ] Confidence source tracking
- [ ] Decay curve configuration

### 4.4 Execution with Rollback
**Source:** LisaSimpson (`deliberative_agent/execution.py`)
**Impact:** Low - advanced error recovery

Features to integrate:
- [ ] Step-by-step execution with verification
- [ ] Rollback capabilities
- [ ] Failure diagnosis
- [ ] Recovery strategies

---

## Unique Algorithms to Preserve

| Algorithm | Source | Purpose | Priority |
|-----------|--------|---------|----------|
| Memory consolidation | TinyTroupe | Long-running agents | High |
| Semantic caching | TinyTroupe | Cost reduction | High |
| GOAP A* planning | LisaSimpson | Intelligent planning | High |
| Anonymous peer review | llm-council | Bias reduction | Medium |
| Multi-round refinement | llm-council | Quality improvement | Medium |
| Confidence provenance | LisaSimpson | Uncertainty tracking | Medium |
| Swarm organization | hivey | Scalable coordination | Low |
| Cost-aware routing | hivey | Resource optimization | Medium |

---

## Integration Statistics

| Metric | Current | Goal |
|--------|---------|------|
| Total repo code | ~10,000 LOC | - |
| Integrated code | ~3,000 LOC | ~7,000 LOC |
| Integration % | ~30% | ~70% |
| Test coverage | Good | Comprehensive |
| Production ready | Partial | Full |

---

## Next Steps (Immediate)

1. **Test the improvements made today:**
   - Retry logic for rate limiting
   - Improved calculator with math functions
   - Synchronous streaming support
   - Loop detection module

2. **Priority integrations:**
   - Semantic caching (reduces API costs)
   - Advanced memory consolidation (enables long sessions)
   - GOAP planning (enables complex reasoning)

3. **Production hardening:**
   - Comprehensive error handling
   - Monitoring and alerting
   - API service for deployment
