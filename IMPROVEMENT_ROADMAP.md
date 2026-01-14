# Club Harness Improvement Roadmap

## Current State (Updated)

**Integrated Components:**
- Core agent architecture with persona system
- OpenRouter backend with 400+ model access
- LLM router with tier-based selection and **retry logic with exponential backoff**
- Memory system (episodic, semantic, lessons, working memory)
- Tool system with **enhanced calculator** (math functions), web search, shell tools
- Council consensus with **3 strategies** (simple ranking, weighted voting, multi-round)
- **Loop detection** (from TinyTroupe) - prevents infinite agent loops
- Streaming support (sync and async)
- **Error handling framework** with custom exceptions and decorators

**Test Coverage:**
- Basic integration tests (6/6 passing)
- Advanced feature tests (5/5 passing)
- **Advanced council tests (3/3 passing)** - multi-round deliberation, confidence tracking, similarity detection
- Comprehensive test suite

**Recent Additions (This Session):**
- [x] Retry logic for rate limiting (exponential backoff)
- [x] Synchronous streaming support
- [x] Enhanced calculator with math functions (abs, sqrt, sin, cos, log, etc.)
- [x] Loop detection module from TinyTroupe
- [x] Error handling framework from hivey
- [x] Multi-round deliberation strategy
- [x] Weighted voting strategy
- [x] Advanced council test suite

---

## Phase 1: Foundation (High Priority)

### 1.1 Advanced Memory Consolidation
**Source:** TinyTroupe (`agent/memory.py` - 49KB)
**Impact:** High - enables long-running agents
**Status:** Not started

Features to integrate:
- [ ] Fixed prefix + lookback window pattern
- [ ] Automatic memory consolidation
- [ ] Reflection consolidation
- [ ] Memory compression for old episodes
- [ ] Configurable consolidation windows

### 1.2 Semantic Caching
**Source:** TinyTroupe (`caching/semantic_cache.py`)
**Impact:** High - reduces API costs, improves performance
**Status:** Reviewed, ready to integrate

Test results show 52-71% similarity between semantically similar queries, validating the caching approach.

Features to integrate:
- [ ] Embedding-based fuzzy cache hits
- [ ] Similarity threshold matching (0.85 default)
- [ ] Hybrid exact + semantic lookup
- [ ] Cache statistics and monitoring
- [ ] LRU eviction for size management

### 1.3 GOAP-Style Planning
**Source:** LisaSimpson (`deliberative_agent/planning.py`)
**Impact:** High - enables deliberative reasoning
**Status:** Reviewed, architecture understood

Key components:
- [ ] A* pathfinding for action sequencing
- [ ] Plan class with verification points
- [ ] Action preconditions and effects (fact-based)
- [ ] Cost estimation and confidence tracking
- [ ] HierarchicalPlanner for complex goals

---

## Phase 2: Features (Medium Priority)

### 2.1 Advanced Consensus Strategies
**Source:** llm-council (`backend/strategies/`)
**Impact:** Medium - improves answer quality
**Status:** PARTIALLY COMPLETE

Completed:
- [x] SimpleRankingStrategy (anonymous peer review)
- [x] WeightedVotingStrategy (performance-based)
- [x] MultiRoundStrategy (iterative refinement, 2-5 rounds)

Remaining:
- [ ] ReasoningAwareStrategy (special handling for o1/o3 models)
- [ ] StrategyRecommender (auto-select based on query)
- [ ] Query classifier for automatic routing
- [ ] Analytics engine for tracking model performance

### 2.2 Verification Framework
**Source:** LisaSimpson (`deliberative_agent/verification.py`)
**Impact:** Medium - enables quality checks
**Status:** Not started

Features to integrate:
- [ ] TypeCheck, TestCheck, SemanticCheck classes
- [ ] Async verification support
- [ ] Confidence-based threshold evaluation
- [ ] Verification result aggregation

### 2.3 Sandbox Manager
**Source:** Agentic-Hub (`harness/sandbox/sandbox_manager.py` - 25KB)
**Impact:** High - essential for safe execution
**Status:** Not started

Features to integrate:
- [ ] SandboxType enum (PRIVATE, SHARED, READONLY, EPHEMERAL)
- [ ] IsolationLevel (NONE, PROCESS, CONTAINER, VM)
- [ ] ResourceLimits (CPU, memory, disk, network)
- [ ] SandboxSnapshot for point-in-time recovery
- [ ] File manifest tracking

### 2.4 Knowledge Base System
**Source:** hivey (SQLite-backed)
**Impact:** Medium - persistent learning
**Status:** Not started

Features to integrate:
- [ ] Vector embedding storage (SQLite + numpy)
- [ ] Semantic similarity search
- [ ] Experience retrieval
- [ ] Cross-session learning

---

## Phase 3: Production Features

### 3.1 Error Handling Framework
**Source:** hivey (`error_handling.py`)
**Impact:** Medium - production robustness
**Status:** COMPLETE

Integrated:
- [x] Custom exception hierarchy (ClubHarnessError, LLMError, RateLimitError, etc.)
- [x] ErrorDetails dataclass
- [x] safe_execute decorator
- [x] retry_on_failure decorator
- [x] validate_input decorator
- [x] handle_llm_errors decorator
- [x] ErrorBoundary context manager

### 3.2 Advanced CLI
**Source:** Agentic-Hub (`harness/cli.py` - 34KB)
**Impact:** Medium - better UX
**Status:** Not started

Features to integrate:
- [ ] Interactive REPL mode
- [ ] Server mode with proper shutdown
- [ ] Rich terminal UI
- [ ] Provider/model listing
- [ ] Command history

### 3.3 Profiling & Monitoring
**Source:** TinyTroupe (`profiling.py`, `monitoring/memory_monitor.py`)
**Impact:** Medium - observability
**Status:** Not started

Features to integrate:
- [ ] MemoryMonitor with alerting
- [ ] MemoryProfiler for detailed analysis
- [ ] Performance telemetry
- [ ] Abnormal growth detection

### 3.4 FastAPI Service
**Source:** hivey (`swarm_service.py`)
**Impact:** Medium - deployment ready
**Status:** Not started

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

### 4.2 Confidence Decay System
**Source:** LisaSimpson
**Impact:** Medium - better uncertainty modeling
**Status:** Basic lesson decay exists, needs enhancement

Features to integrate:
- [ ] Time-based confidence decay (half-life model)
- [ ] Confidence source tracking (VERIFICATION, INFERENCE, USER_FEEDBACK)
- [ ] Decay curve configuration
- [ ] Confidence aggregation for multi-source facts

### 4.3 Execution with Rollback
**Source:** LisaSimpson (`deliberative_agent/execution.py`)
**Impact:** Low - advanced error recovery

Features to integrate:
- [ ] Step-by-step execution with verification
- [ ] Rollback capabilities
- [ ] Failure diagnosis
- [ ] Recovery strategies

---

## Unique Algorithms to Preserve

| Algorithm | Source | Purpose | Status |
|-----------|--------|---------|--------|
| Memory consolidation | TinyTroupe | Long-running agents | Not started |
| Semantic caching | TinyTroupe | Cost reduction | Ready to integrate |
| Loop detection | TinyTroupe | Agent safety | **COMPLETE** |
| GOAP A* planning | LisaSimpson | Intelligent planning | Reviewed |
| Anonymous peer review | llm-council | Bias reduction | **COMPLETE** |
| Multi-round refinement | llm-council | Quality improvement | **COMPLETE** |
| Weighted voting | llm-council | Performance-based consensus | **COMPLETE** |
| Confidence provenance | LisaSimpson | Uncertainty tracking | Partial |
| Error handling | hivey | Production robustness | **COMPLETE** |
| Retry with backoff | hivey | Rate limit handling | **COMPLETE** |

---

## Integration Statistics

| Metric | Previous | Current | Goal |
|--------|----------|---------|------|
| Total repo code | ~10,000 LOC | ~10,000 LOC | - |
| Integrated code | ~3,000 LOC | ~4,500 LOC | ~7,000 LOC |
| Integration % | ~30% | ~45% | ~70% |
| Test coverage | Good | Excellent | Comprehensive |
| Production ready | Partial | Good | Full |
| Council strategies | 1 | 3 | 5 |

---

## Testing Results Summary

### Free Models Tested:
- `meta-llama/llama-3.2-3b-instruct:free` - Reliable, good quality
- `google/gemma-3n-e2b-it:free` - Fast responses
- `nvidia/nemotron-nano-9b-v2:free` - Works well
- `moonshotai/kimi-k2:free` - Rate limited frequently
- `qwen/qwen3-coder:free` - Occasional 500 errors

### Key Findings:
1. **Multi-round deliberation improves consensus** - Models refine answers based on peer context
2. **Confidence tracking works** - Models correctly assess certainty levels
3. **Response similarity validates caching** - 52-71% similarity for semantic queries
4. **Rate limiting is common** - Retry logic is essential for free models

---

## Next Steps (Recommended Priority)

### Immediate (High Value):
1. **Semantic Caching** - Test results validate approach, significant cost savings
2. **Memory Consolidation** - Enables practical long-running agents
3. **Analytics Engine** - Track model performance for weighted voting

### Short-term:
4. **GOAP Planning** - Core differentiator for deliberative agents
5. **Verification Framework** - Quality assurance for agent outputs
6. **FastAPI Service** - Production deployment

### Medium-term:
7. **Sandbox Manager** - Safe code execution
8. **Knowledge Base** - Persistent learning
9. **Advanced CLI** - Better developer UX

---

## Architecture Decisions

### Why Multi-Round Deliberation?
Testing showed that when models see top-ranked responses from Round 1, they:
- Incorporate valid points from others
- Strengthen their reasoning
- Converge toward better answers
- Chairman synthesis benefits from evolved context

### Why Error Handling Framework?
Free models frequently:
- Hit rate limits (429 errors)
- Return server errors (500)
- Timeout on complex queries
The retry_on_failure decorator and ErrorBoundary make the system robust.

### Why Loop Detection?
Without it, agents can:
- Repeat the same action indefinitely
- Oscillate between two states
- Waste API calls on unproductive patterns
The AdvancedLoopDetector catches 5 types of loops.
