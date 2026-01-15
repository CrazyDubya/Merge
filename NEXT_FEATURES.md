# Next Features to Integrate

These 5 high-value features were identified from source repos but not yet integrated into Club Harness. They are ready for implementation in a follow-up PR.

---

## 1. Self-Evaluation Flywheel System (HIGH PRIORITY)

**Source**: `repos/Agentic-Hub/harness/evaluation/self_eval.py`

**What it does**: Creates a continuous improvement loop where agents evaluate their own outputs, identify weaknesses, and generate training data to improve.

**Key components**:
- `SelfEvaluator` class that scores agent responses on multiple dimensions
- Automated weakness detection and categorization
- Training example generation from successful/failed interactions
- Flywheel metrics tracking improvement over time

**Implementation approach**:
1. Create `club_harness/evaluation/self_eval.py`
2. Integrate with existing verification framework (`verification/checks.py`)
3. Add hooks in `Agent.chat()` to optionally run self-evaluation
4. Store evaluation results in memory system for learning

**Why high priority**: Enables continuous improvement without manual intervention. Direct synergy with existing verification and memory systems.

---

## 2. Demographic Persona Generation (MEDIUM PRIORITY)

**Source**: `repos/TinyTroupe/tinytroupe/persona_generator.py`

**What it does**: Generates realistic agent personas with demographic attributes, personality traits, and behavioral patterns for simulation and testing.

**Key components**:
- `PersonaGenerator` with demographic distributions
- Personality trait mapping (Big Five model)
- Behavioral tendency generation
- Persona consistency validation

**Implementation approach**:
1. Create `club_harness/personas/generator.py`
2. Integrate with Agent's system prompt construction
3. Add persona presets for common use cases
4. Support persona persistence across sessions

**Why medium priority**: Useful for testing, simulation, and creating diverse agent behaviors. Not critical for core functionality.

---

## 3. Training Data Generation (MEDIUM PRIORITY)

**Source**: `repos/Agentic-Hub/harness/training/data_generator.py`

**What it does**: Automatically generates high-quality training data from agent interactions for fine-tuning or RLHF.

**Key components**:
- Conversation filtering (quality thresholds)
- Format conversion (OpenAI, Anthropic, custom)
- Deduplication and diversity scoring
- Annotation helpers for human review

**Implementation approach**:
1. Create `club_harness/training/data_generator.py`
2. Hook into conversation history in `Agent`
3. Add export formats for major training pipelines
4. Integrate with self-eval to auto-label quality

**Why medium priority**: Valuable for teams doing fine-tuning. Builds on self-eval system.

---

## 4. Streaming & Tool Call Collection (MEDIUM PRIORITY)

**Source**: `repos/qwen-code/qwen_agent/llm/streaming.py`

**What it does**: Handles streaming responses with proper tool call extraction and partial response handling.

**Key components**:
- `StreamingHandler` for chunked responses
- Tool call accumulator (handles split JSON)
- Progress callbacks for UI integration
- Timeout and retry logic for streams

**Implementation approach**:
1. Extend `club_harness/llm/openrouter.py` with streaming support
2. Add `stream=True` option to `OpenRouterClient.chat()`
3. Implement tool call parser that works on partial data
4. Add callback hooks for progress reporting

**Why medium priority**: Better UX for long responses. Some models perform better with streaming.

---

## 5. Knowledge Base with Semantic Search (MEDIUM PRIORITY)

**Source**: `repos/hivey/hivey/knowledge/semantic_kb.py`

**What it does**: Provides a semantic knowledge base that agents can query using natural language, with vector similarity search.

**Key components**:
- `SemanticKnowledgeBase` with embedding support
- Document chunking and indexing
- Similarity search with configurable thresholds
- Source attribution and citation

**Implementation approach**:
1. Create `club_harness/knowledge/semantic_kb.py`
2. Support pluggable embedding backends (OpenAI, local)
3. Integrate with Agent's context building
4. Add RAG (Retrieval Augmented Generation) helper

**Why medium priority**: Powerful for domain-specific agents. Requires embedding API setup.

---

## Implementation Order Recommendation

```
1. Self-Evaluation Flywheel  (builds on verification, enables #3)
   └── 2. Training Data Generation (uses self-eval scores)

3. Streaming & Tool Calls    (independent, improves UX)

4. Knowledge Base            (independent, enables RAG)
   └── 5. Persona Generation (can use KB for persona details)
```

## Quick Start for Next Session

```bash
# View source implementations
cat repos/Agentic-Hub/harness/evaluation/self_eval.py
cat repos/TinyTroupe/tinytroupe/persona_generator.py
cat repos/qwen-code/qwen_agent/llm/streaming.py

# Run existing tests to verify nothing is broken
python test_complex_integration.py
python test_baseline_problems.py --quick

# Start with Self-Eval (highest value)
# Create: club_harness/evaluation/self_eval.py
```

## Current State Reference

- **Branch**: `claude/test-openrouter-integration-2woKZ`
- **Last commit**: `eb3408a Add baseline problem set for evaluating LLM capabilities`
- **Tests passing**: 8/8 complex integration, 11/12 baseline problems
- **Model in use**: `google/gemma-3n-e2b-it:free` (OpenRouter)
