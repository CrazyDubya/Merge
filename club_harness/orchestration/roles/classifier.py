"""Task classifier: torch/transformers when available, heuristic fallback otherwise.

ChipCliff's pm_algorithm.py classified tasks with a DistilBERT sequence
classifier downloaded at import time - heavy and import-blocking. This port
keeps the same interface (classify_task -> "coding" | "research" | None) but:

- The torch path is lazy: only attempted inside classify_task(), never at import.
- If torch/transformers (or the model) is unavailable, a keyword heuristic
  decides instead. No heavy deps in Merge core.

Ported from role-based-llm-framework/pm_algorithm.py via git subtree merge;
see docs/PROVENANCE.md.
"""

import os
from typing import Optional

MODEL_DIR = "models/task_classifier"
MODEL_NAME = "distilbert-base-uncased"

CODING_KEYWORDS = {
    "code", "coding", "program", "programming", "function", "class", "api",
    "bug", "debug", "script", "app", "website", "html", "css", "javascript",
    "python", "sql", "database", "algorithm", "implement", "refactor",
    "deploy", "test", "pytest", "compile", "sdk", "library", "endpoint",
}

RESEARCH_KEYWORDS = {
    "research", "analyze", "analysis", "compare", "comparison", "survey",
    "report", "paper", "study", "investigate", "explore", "summarize",
    "summary", "find", "search", "what is", "who is", "history of",
    "explain", "overview", "trends", "news",
}


def classify_task_heuristic(task: str) -> Optional[str]:
    """Keyword-based task classification. No ML dependencies."""
    lowered = task.lower()
    coding_score = sum(1 for kw in CODING_KEYWORDS if kw in lowered)
    research_score = sum(1 for kw in RESEARCH_KEYWORDS if kw in lowered)
    if coding_score == 0 and research_score == 0:
        return None
    return "coding" if coding_score >= research_score else "research"


class TaskClassifier:
    """Classifies tasks as 'coding' or 'research'.

    Tries the torch/transformers DistilBERT path first (matching ChipCliff's
    original behavior); falls back to the keyword heuristic when torch,
    transformers, or the model weights are unavailable.
    """

    def __init__(self, model_dir: str = MODEL_DIR, use_torch: bool = True) -> None:
        self.model_dir = model_dir
        self.use_torch = use_torch
        self._torch_classifier = None  # lazy

    def _load_torch_classifier(self):
        """Load (downloading once if needed) the DistilBERT classifier."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(f"torch/transformers not installed: {e}")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model.save_pretrained(self.model_dir)
            tokenizer.save_pretrained(self.model_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._torch_classifier = (model, tokenizer)
        return self._torch_classifier

    def classify_task(self, task: str) -> Optional[str]:
        """Classify a task. Returns 'coding', 'research', or None."""
        if self.use_torch:
            try:
                import torch

                model, tokenizer = (
                    self._torch_classifier or self._load_torch_classifier()
                )
                inputs = tokenizer(task, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                return "coding" if prediction == 0 else "research"
            except Exception:
                # Any failure (no torch, no model, no network) -> heuristic.
                pass
        return classify_task_heuristic(task)
