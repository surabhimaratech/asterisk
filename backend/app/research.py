from dataclasses import dataclass
from typing import Protocol

from .ai import OpenRouterClient


@dataclass(frozen=True)
class RetrievedSource:
    url: str
    title: str
    excerpt: str


@dataclass(frozen=True)
class ResearchPacket:
    synthesis: str
    sources: list[RetrievedSource]


class ResearchProvider(Protocol):
    def research(self, item: dict[str, object]) -> ResearchPacket: ...
    def research_problems(self, item: dict[str, object], problems: list[dict[str, object]]) -> ResearchPacket: ...
    def recover_candidate(self, item: dict[str, object], candidate: dict[str, object]) -> ResearchPacket: ...


class OpenRouterResearchProvider:
    """OpenRouter web search adapter. Provider details stay out of routes and persistence."""

    def __init__(self, client: OpenRouterClient | None = None) -> None:
        self.client = client or OpenRouterClient()

    def research(self, item: dict[str, object]) -> ResearchPacket:
        synthesis, citations = self.client.research_item(item)
        sources = [RetrievedSource(**citation) for citation in citations]
        return ResearchPacket(synthesis=synthesis, sources=sources)

    def research_problems(self, item: dict[str, object], problems: list[dict[str, object]]) -> ResearchPacket:
        synthesis, citations = self.client.research_problems(item, problems)
        return ResearchPacket(synthesis=synthesis, sources=[RetrievedSource(**citation) for citation in citations])

    def recover_candidate(self, item: dict[str, object], candidate: dict[str, object]) -> ResearchPacket:
        synthesis, citations = self.client.research_candidate_evidence(candidate)
        return ResearchPacket(synthesis=synthesis, sources=[RetrievedSource(**citation) for citation in citations])


def get_research_provider() -> ResearchProvider:
    return OpenRouterResearchProvider()
