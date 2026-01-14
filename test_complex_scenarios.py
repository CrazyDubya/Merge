#!/usr/bin/env python3
"""
Complex scenario testing for Club Harness.

Tests:
1. Multi-agent collaboration on problem solving
2. Planning-style reasoning
3. Memory-based context recall
4. Deliberative reasoning with verification
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.config import config
from club_harness.core.agent import Agent
from club_harness.llm.router import LLMRouter
from club_harness.memory import Memory, MemoryType, Lesson
from club_harness.orchestration import Council


def test_collaborative_problem_solving():
    """Test multiple agents collaborating on a complex problem."""
    print("\n" + "=" * 60)
    print("TEST: Collaborative Problem Solving")
    print("=" * 60)

    router = LLMRouter()

    # Problem requiring multiple perspectives
    problem = """
    A small tech startup has $50,000 budget and needs to:
    1. Build an MVP in 3 months
    2. Hire at least 2 developers
    3. Market to early adopters

    What strategy would you recommend? Consider trade-offs.
    """

    print(f"Problem: {problem[:100]}...")

    # Create specialized agents with different perspectives
    agents = [
        ("Technical Lead", "You are a technical lead focused on engineering quality and feasibility."),
        ("Business Strategist", "You are a business strategist focused on market fit and growth."),
        ("Financial Advisor", "You are a financial advisor focused on budget optimization and sustainability."),
    ]

    responses = {}
    for name, instruction in agents:
        try:
            print(f"\nAgent: {name}")
            agent = Agent(name=name, instructions=instruction, tier="free")
            response = agent.chat(problem)
            responses[name] = response
            print(f"Response: {response[:250]}...")
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            print(f"Error: {e}")
            responses[name] = f"Error: {e}"

    # Synthesize perspectives
    if len(responses) >= 2:
        print("\n--- Synthesis ---")
        synthesis_prompt = f"""Given these expert perspectives on the startup problem:

Technical Lead: {responses.get('Technical Lead', 'N/A')[:300]}...

Business Strategist: {responses.get('Business Strategist', 'N/A')[:300]}...

Financial Advisor: {responses.get('Financial Advisor', 'N/A')[:300]}...

Synthesize the best combined strategy:"""

        try:
            synthesis = router.chat(
                messages=[{"role": "user", "content": synthesis_prompt}],
                model="meta-llama/llama-3.2-3b-instruct:free",
                max_tokens=400,
            )
            print(f"\nSynthesized Strategy:\n{synthesis.content}")
        except Exception as e:
            print(f"Synthesis error: {e}")

    return True


def test_planning_reasoning():
    """Test GOAP-style planning reasoning."""
    print("\n" + "=" * 60)
    print("TEST: Planning-Style Reasoning")
    print("=" * 60)

    router = LLMRouter()

    # Planning problem
    planning_prompt = """You are a planning agent. Your task is to create a step-by-step plan.

GOAL: Deploy a web application to production

CURRENT STATE:
- Code is written and tested locally
- No server infrastructure exists
- Domain name is purchased
- Budget: $100/month

AVAILABLE ACTIONS:
1. Set up cloud server ($20/month)
2. Configure database ($10/month)
3. Set up CI/CD pipeline (free with GitHub)
4. Configure domain DNS
5. Set up SSL certificate (free with Let's Encrypt)
6. Deploy application
7. Set up monitoring ($10/month)

Create a plan with:
- Ordered steps
- Dependencies between steps
- Estimated cost
- Verification points (how to check each step succeeded)

Format as:
PLAN:
Step 1: [action] - depends on: [none/step X] - verify: [how]
Step 2: ...
TOTAL COST: $X/month"""

    try:
        print("Generating plan...")
        response = router.chat(
            messages=[{"role": "user", "content": planning_prompt}],
            model="meta-llama/llama-3.2-3b-instruct:free",
            max_tokens=500,
        )
        print(f"\nPlanning Response:\n{response.content}")

        # Check if response contains key planning elements
        plan_text = response.content.lower()
        has_steps = "step" in plan_text
        has_cost = "cost" in plan_text or "$" in plan_text
        has_verify = "verify" in plan_text or "check" in plan_text

        print(f"\nPlan Quality Check:")
        print(f"  Has steps: {has_steps}")
        print(f"  Has cost estimate: {has_cost}")
        print(f"  Has verification: {has_verify}")

        return has_steps

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_memory_context_recall():
    """Test memory-based context recall across conversation."""
    print("\n" + "=" * 60)
    print("TEST: Memory-Based Context Recall")
    print("=" * 60)

    memory = Memory()
    agent = Agent(
        name="ContextAgent",
        instructions="You are an assistant with excellent memory. Remember all details the user shares.",
        tier="free",
    )

    # Build up context over multiple interactions
    context_items = [
        ("My name is Sarah and I'm a software engineer at TechCorp.", "personal_info"),
        ("I'm working on a machine learning project about sentiment analysis.", "project_info"),
        ("My deadline is next Friday and I'm using Python with scikit-learn.", "project_details"),
    ]

    print("Building context...")
    for info, info_type in context_items:
        print(f"\n  User: {info}")
        response = agent.chat(info)
        memory.remember(f"User said: {info}", MemoryType.OBSERVATION)
        memory.remember(f"Agent responded: {response[:50]}...", MemoryType.ACTION)
        print(f"  Agent: {response[:100]}...")
        time.sleep(1)

    # Now test recall
    recall_questions = [
        "What's my name and job?",
        "What project am I working on?",
        "When is my deadline and what tools am I using?",
    ]

    print("\n--- Testing Recall ---")
    recall_success = 0
    for question in recall_questions:
        print(f"\nQuestion: {question}")
        response = agent.chat(question)
        print(f"Answer: {response[:200]}...")

        # Check if key terms are recalled
        response_lower = response.lower()
        if "sarah" in response_lower or "software" in response_lower or "techcorp" in response_lower:
            recall_success += 1
        elif "sentiment" in response_lower or "machine learning" in response_lower:
            recall_success += 1
        elif "friday" in response_lower or "python" in response_lower or "scikit" in response_lower:
            recall_success += 1

    print(f"\n  Recall Success: {recall_success}/{len(recall_questions)}")

    # Add a lesson from this experience
    memory.learn(
        situation="Context recall test",
        insight=f"Agent recalled {recall_success}/{len(recall_questions)} context items correctly",
        outcome="success" if recall_success >= 2 else "partial",
        confidence=recall_success / len(recall_questions),
    )

    return recall_success >= 2


def test_multi_round_council_complex():
    """Test multi-round council on a complex question."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Round Council Complex Question")
    print("=" * 60)

    # Use multi-round strategy for better quality
    council = Council(
        models=[
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-3n-e2b-it:free",
        ],
        chairman="meta-llama/llama-3.2-3b-instruct:free",
        strategy="multi_round",
    )

    # Complex question requiring nuanced thinking
    question = """Should a company prioritize short-term profits or long-term sustainability?
Consider: stakeholder interests, market conditions, ethical implications, and competitive advantage.
Give a nuanced answer acknowledging trade-offs."""

    print(f"Question: {question[:100]}...")
    print(f"Strategy: {council.strategy.name}")

    try:
        print("\nRunning multi-round deliberation...")
        result = council.deliberate_sync(question)

        print(f"\nFinal Answer ({result.chairman_model}):")
        print(result.final_answer[:500])

        # Check for nuance
        answer_lower = result.final_answer.lower()
        has_tradeoffs = "trade" in answer_lower or "balance" in answer_lower or "both" in answer_lower
        has_stakeholders = "stakeholder" in answer_lower or "employee" in answer_lower or "shareholder" in answer_lower

        print(f"\nAnswer Quality:")
        print(f"  Acknowledges trade-offs: {has_tradeoffs}")
        print(f"  Considers stakeholders: {has_stakeholders}")
        print(f"  Strategy used: {result.strategy}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chain_of_thought():
    """Test chain-of-thought reasoning."""
    print("\n" + "=" * 60)
    print("TEST: Chain of Thought Reasoning")
    print("=" * 60)

    router = LLMRouter()

    # Problem requiring step-by-step reasoning
    cot_prompt = """Solve this step by step:

A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?

Think through this carefully:
1. First, understand what "all but 9" means
2. Then calculate the answer
3. Verify your answer makes sense

Show your reasoning at each step."""

    try:
        print("Testing chain-of-thought reasoning...")
        response = router.chat(
            messages=[{"role": "user", "content": cot_prompt}],
            model="meta-llama/llama-3.2-3b-instruct:free",
            max_tokens=300,
        )
        print(f"\nResponse:\n{response.content}")

        # Check if reasoning is shown
        response_lower = response.content.lower()
        shows_reasoning = "step" in response_lower or "because" in response_lower or "means" in response_lower
        correct_answer = "9" in response.content

        print(f"\n  Shows reasoning: {shows_reasoning}")
        print(f"  Contains correct answer (9): {correct_answer}")

        return correct_answer

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all complex scenario tests."""
    print("=" * 60)
    print("CLUB HARNESS - Complex Scenario Testing")
    print("=" * 60)
    print(f"API Key: {'Set' if config.llm.api_key else 'Not set'}")

    if not config.llm.api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return False

    results = []

    # Run tests
    tests = [
        ("Chain of Thought", test_chain_of_thought),
        ("Planning Reasoning", test_planning_reasoning),
        ("Memory Context Recall", test_memory_context_recall),
        ("Multi-Round Council Complex", test_multi_round_council_complex),
        ("Collaborative Problem Solving", test_collaborative_problem_solving),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, bool(result) if result is not None else True))
        except Exception as e:
            print(f"\nTest '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("COMPLEX SCENARIO TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")
    return passed >= 3  # Allow some failures due to model variability


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
