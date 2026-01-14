"""
Council consensus system for multi-LLM deliberation.

Implements patterns from llm-council:
- Anonymous peer review
- Multiple consensus strategies
- Chairman synthesis
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..llm.router import LLMRouter, LLMResponse, router


@dataclass
class CouncilResponse:
    """Response from a single council member."""
    model: str
    content: str
    label: str  # Anonymous label (A, B, C, etc.)
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CouncilRanking:
    """Ranking from a single council member."""
    model: str
    rankings: List[str]  # Ordered list of labels
    reasoning: Optional[str] = None


@dataclass
class CouncilResult:
    """Result of council deliberation."""
    query: str
    final_answer: str
    stage1_responses: List[CouncilResponse]
    stage2_rankings: List[CouncilRanking]
    stage3_synthesis: str
    aggregate_rankings: Dict[str, float]  # model -> avg rank
    chairman_model: str
    strategy: str
    total_tokens: int = 0


class ConsensusStrategy(ABC):
    """Abstract base class for consensus strategies."""

    name: str = "base"

    @abstractmethod
    async def deliberate(
        self,
        query: str,
        council_models: List[str],
        chairman_model: str,
        router: LLMRouter,
    ) -> CouncilResult:
        """Run the deliberation process."""
        pass


class SimpleRankingStrategy(ConsensusStrategy):
    """
    3-stage deliberation with anonymous peer review.

    Stage 1: All models respond independently
    Stage 2: Each model ranks all responses anonymously
    Stage 3: Chairman synthesizes final answer
    """

    name = "simple_ranking"

    async def deliberate(
        self,
        query: str,
        council_models: List[str],
        chairman_model: str,
        router: LLMRouter,
    ) -> CouncilResult:
        # Stage 1: Get individual opinions
        stage1 = await self._stage1_opinions(query, council_models, router)

        # Stage 2: Anonymous peer ranking
        stage2, label_to_model = await self._stage2_rankings(
            query, stage1, council_models, router
        )

        # Calculate aggregate rankings
        aggregate = self._calculate_aggregate_rankings(stage2, label_to_model)

        # Stage 3: Chairman synthesis
        stage3 = await self._stage3_synthesis(
            query, stage1, stage2, aggregate, chairman_model, router
        )

        total_tokens = sum(r.tokens_used for r in stage1) + stage3.total_tokens

        return CouncilResult(
            query=query,
            final_answer=stage3.content,
            stage1_responses=stage1,
            stage2_rankings=stage2,
            stage3_synthesis=stage3.content,
            aggregate_rankings=aggregate,
            chairman_model=chairman_model,
            strategy=self.name,
            total_tokens=total_tokens,
        )

    async def _stage1_opinions(
        self,
        query: str,
        models: List[str],
        router: LLMRouter,
    ) -> List[CouncilResponse]:
        """Stage 1: Get independent opinions from all models."""
        responses = []
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Query all models (could be parallelized with asyncio.gather)
        for i, model in enumerate(models):
            try:
                result = router.chat(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    temperature=0.7,
                )
                responses.append(CouncilResponse(
                    model=model,
                    content=result.content,
                    label=labels[i],
                    tokens_used=result.total_tokens,
                ))
            except Exception as e:
                # Graceful degradation
                responses.append(CouncilResponse(
                    model=model,
                    content=f"[Error: {str(e)}]",
                    label=labels[i],
                ))

        return responses

    async def _stage2_rankings(
        self,
        query: str,
        responses: List[CouncilResponse],
        models: List[str],
        router: LLMRouter,
    ) -> Tuple[List[CouncilRanking], Dict[str, str]]:
        """Stage 2: Each model ranks all responses anonymously."""
        # Build anonymous response list
        label_to_model = {r.label: r.model for r in responses}

        anonymous_responses = "\n\n".join([
            f"Response {r.label}:\n{r.content}"
            for r in responses
        ])

        ranking_prompt = f"""You are evaluating multiple AI responses to this query:

Query: {query}

Here are the anonymous responses:

{anonymous_responses}

Please rank these responses from best to worst. Consider:
- Accuracy and correctness
- Completeness
- Clarity and helpfulness

Provide your ranking in exactly this format:
FINAL RANKING:
1. [letter]
2. [letter]
...

Then briefly explain your reasoning."""

        rankings = []
        for model in models:
            try:
                result = router.chat(
                    messages=[{"role": "user", "content": ranking_prompt}],
                    model=model,
                    temperature=0.3,  # Lower temp for evaluation
                    max_tokens=500,
                )

                # Parse ranking
                parsed = self._parse_ranking(result.content, len(responses))
                rankings.append(CouncilRanking(
                    model=model,
                    rankings=parsed,
                    reasoning=result.content,
                ))
            except Exception:
                # Skip failed rankings
                pass

        return rankings, label_to_model

    def _parse_ranking(self, text: str, num_responses: int) -> List[str]:
        """Parse ranking from model output."""
        # Look for FINAL RANKING section
        match = re.search(r"FINAL RANKING:\s*\n((?:\d+\.\s*[A-Z]\s*\n?)+)", text, re.IGNORECASE)
        if match:
            ranking_text = match.group(1)
            # Extract letters
            letters = re.findall(r"\d+\.\s*([A-Z])", ranking_text, re.IGNORECASE)
            return [l.upper() for l in letters[:num_responses]]

        # Fallback: find any ordered list of letters
        letters = re.findall(r"([A-Z])\s*[,>]?\s*", text.upper())
        seen = set()
        result = []
        for l in letters:
            if l not in seen and l in "ABCDEFGHIJ":
                seen.add(l)
                result.append(l)
        return result[:num_responses]

    def _calculate_aggregate_rankings(
        self,
        rankings: List[CouncilRanking],
        label_to_model: Dict[str, str],
    ) -> Dict[str, float]:
        """Calculate average rank for each model."""
        model_positions: Dict[str, List[int]] = {m: [] for m in label_to_model.values()}

        for ranking in rankings:
            for position, label in enumerate(ranking.rankings, start=1):
                if label in label_to_model:
                    model = label_to_model[label]
                    model_positions[model].append(position)

        # Calculate averages
        result = {}
        for model, positions in model_positions.items():
            if positions:
                result[model] = sum(positions) / len(positions)
            else:
                result[model] = float('inf')

        return result

    async def _stage3_synthesis(
        self,
        query: str,
        responses: List[CouncilResponse],
        rankings: List[CouncilRanking],
        aggregate: Dict[str, float],
        chairman_model: str,
        router: LLMRouter,
    ) -> LLMResponse:
        """Stage 3: Chairman synthesizes final answer."""
        # Sort responses by aggregate ranking
        sorted_responses = sorted(
            responses,
            key=lambda r: aggregate.get(r.model, float('inf'))
        )

        responses_text = "\n\n".join([
            f"Response from {r.model} (avg rank: {aggregate.get(r.model, 'N/A'):.1f}):\n{r.content}"
            for r in sorted_responses
        ])

        synthesis_prompt = f"""You are the chairman of an AI council. Multiple AI models have responded to a query, and the council has ranked their responses.

Original Query: {query}

Council Responses (ordered by consensus ranking):

{responses_text}

As chairman, synthesize the best possible answer by:
1. Taking the strongest points from top-ranked responses
2. Correcting any errors identified by other responses
3. Providing a comprehensive, accurate final answer

Your synthesized answer:"""

        return router.chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=chairman_model,
            temperature=0.5,
        )


class WeightedVotingStrategy(ConsensusStrategy):
    """
    Consensus strategy with performance-based voting weights.

    Models that historically perform well get more influence.
    """

    name = "weighted_voting"

    def __init__(self, model_weights: Optional[Dict[str, float]] = None):
        self.model_weights = model_weights or {}

    async def deliberate(
        self,
        query: str,
        council_models: List[str],
        chairman_model: str,
        router: LLMRouter,
    ) -> CouncilResult:
        # Use simple ranking as base
        simple = SimpleRankingStrategy()
        result = await simple.deliberate(query, council_models, chairman_model, router)

        # Apply weights to rankings
        weighted_rankings = self._apply_weights(result.stage2_rankings)
        result.aggregate_rankings = weighted_rankings
        result.strategy = self.name

        return result

    def _apply_weights(self, rankings: List[CouncilRanking]) -> Dict[str, float]:
        """Apply model weights to rankings."""
        model_scores: Dict[str, float] = {}
        model_counts: Dict[str, float] = {}

        for ranking in rankings:
            weight = self.model_weights.get(ranking.model, 1.0)

            for position, label in enumerate(ranking.rankings, start=1):
                if label not in model_scores:
                    model_scores[label] = 0.0
                    model_counts[label] = 0.0

                model_scores[label] += position * weight
                model_counts[label] += weight

        # Calculate weighted averages
        return {
            label: score / count if count > 0 else float('inf')
            for label, (score, count) in zip(
                model_scores.keys(),
                zip(model_scores.values(), model_counts.values())
            )
        }


class Council:
    """
    Multi-LLM council for consensus-based answers.

    Usage:
        council = Council(models=["model1", "model2", "model3"])
        result = await council.deliberate("What is the meaning of life?")
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        chairman: Optional[str] = None,
        strategy: str = "simple_ranking",
        router: Optional[LLMRouter] = None,
    ):
        # Default free models for testing
        self.models = models or [
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen3-coder:free",
        ]
        self.chairman = chairman or self.models[0]
        self.router = router or LLMRouter()

        # Strategy registry
        self._strategies: Dict[str, ConsensusStrategy] = {
            "simple_ranking": SimpleRankingStrategy(),
            "weighted_voting": WeightedVotingStrategy(),
        }
        self.strategy = self._strategies.get(strategy, SimpleRankingStrategy())

    async def deliberate(self, query: str) -> CouncilResult:
        """Run council deliberation on a query."""
        return await self.strategy.deliberate(
            query=query,
            council_models=self.models,
            chairman_model=self.chairman,
            router=self.router,
        )

    def deliberate_sync(self, query: str) -> CouncilResult:
        """Synchronous wrapper for deliberation."""
        return asyncio.get_event_loop().run_until_complete(self.deliberate(query))

    def quick_consensus(self, query: str) -> str:
        """Quick synchronous consensus - returns just the final answer."""
        result = self.deliberate_sync(query)
        return result.final_answer
