"""Researcher role: build search queries, fetch sources, summarize.

Ports ChipCliff's researcher_algorithm.py. Query enhancement goes through
club_harness's LLMRouter (falling back to base queries when no router/key is
available); web fetching uses requests + BeautifulSoup, both optional and
imported lazily so the role imports cleanly without them.

Ported from role-based-llm-framework/researcher_algorithm.py via git subtree
merge; see docs/PROVENANCE.md.
"""

from typing import Dict, List, Optional

try:
    from ...llm.router import LLMRouter
except ImportError:  # pragma: no cover - defensive
    LLMRouter = None  # type: ignore


class ResearcherRole:
    """Researches a task: queries -> sources -> summary."""

    def __init__(self, router: Optional["LLMRouter"] = None, model: Optional[str] = None):
        if router is None and LLMRouter is not None:
            router = LLMRouter()
        self.router = router
        self.model = model

    def research(self, task: str) -> str:
        """Run the full research pipeline for a task. Returns a summary."""
        base_queries = [f"{task} best practices", f"{task} tutorials"]
        queries = self.enhance_queries(task, base_queries)
        results = self.fetch_data(queries)
        return self.summarize_results(results)

    def enhance_queries(self, task: str, base_queries: List[str]) -> List[str]:
        """Ask the LLM for more targeted queries; fall back to base queries."""
        if self.router is None:
            return base_queries
        try:
            response = self.router.chat(
                messages=[
                    {"role": "user",
                     "content": f"Given the task: '{task}' and base queries: "
                                f"{base_queries}, generate 3 more specific and "
                                f"targeted search queries. One per line, no numbering."},
                ],
                model=self.model,
            )
            extra = [q.strip() for q in response.content.split("\n") if q.strip()]
            return base_queries + extra
        except Exception:
            return base_queries

    def fetch_data(self, queries: List[str]) -> List[Dict[str, str]]:
        """Fetch web results for queries. Requires requests + beautifulsoup4."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        data: List[Dict[str, str]] = []
        for query in queries:
            try:
                url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.find_all("div", class_="tF2Cxc"):
                    title = result.find("h3").text if result.find("h3") else "No Title"
                    desc_el = result.find("div", class_="VwiC3b")
                    snippet = desc_el.text if desc_el else "No Description"
                    data.append({"title": title, "description": snippet})
            except Exception:
                continue
        return data

    def summarize_results(self, data: List[Dict[str, str]]) -> str:
        """Summarize fetched results."""
        if not data:
            return "No results found."
        return "\n".join(
            f"Title: {item['title']}\nDescription: {item['description']}\n"
            for item in data[:5]
        )


__all__ = ["ResearcherRole"]
