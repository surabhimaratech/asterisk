"""Bounded, non-persisting research workflow for Discovery Engine v0.1."""

from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from .ai import OpenRouterClient, ResearchedDiscoveryClaim
from .research import ResearchProvider, RetrievedSource


class NoDiscoveryError(RuntimeError):
    """The bounded experiment completed without enough evidence for a discovery."""


class DiscoveryResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int = Field(gt=0)


class DiscoverySourceRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    url: str
    title: str
    excerpt: str


class ResearchedDiscoveryRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_item_id: int
    seed_title: str
    directions: list[dict[str, object]]
    title: str
    discovery: str
    why_surprising: str
    connection_to_library: str | None
    what_remains_uncertain: str
    next_rabbit_hole: str
    claims: list[ResearchedDiscoveryClaim]
    evidence: list[DiscoverySourceRead]
    evidence_status: str = "Research-backed · cited sources support factual claims."
    persisted: bool = False


def _usable_sources(sources: list[RetrievedSource]) -> list[DiscoverySourceRead]:
    usable: list[DiscoverySourceRead] = []
    seen_urls: set[str] = set()
    for source in sources:
        parsed = urlparse(source.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not source.excerpt.strip():
            continue
        if source.url in seen_urls:
            continue
        seen_urls.add(source.url)
        usable.append(DiscoverySourceRead(
            source_id=f"discovery-source-{len(usable) + 1}", url=source.url,
            title=source.title.strip() or source.url, excerpt=source.excerpt.strip()[:1800],
        ))
    return usable[:8]


def run_researched_discovery(
    seed: dict[str, object], ai: OpenRouterClient, research: ResearchProvider,
) -> ResearchedDiscoveryRead:
    directions = ai.propose_discovery_directions(seed)
    direction_payloads = [direction.model_dump() for direction in directions.directions]
    packet = research.research_discovery(seed, direction_payloads)
    sources = _usable_sources(packet.sources)
    if not sources:
        raise NoDiscoveryError("Research found no inspectable sources strong enough for a discovery.")

    source_payloads = [source.model_dump() for source in sources]
    draft = ai.synthesize_researched_discovery(seed, direction_payloads, packet.synthesis, source_payloads)
    if not draft.accepted:
        raise NoDiscoveryError(draft.rejection_reason or "The retrieved evidence did not support a compelling discovery.")
    required_text = (
        draft.title, draft.discovery, draft.why_surprising,
        draft.what_remains_uncertain, draft.next_rabbit_hole,
    )
    if any(not value.strip() for value in required_text):
        raise NoDiscoveryError("The researched result was incomplete, so Asterisk did not present it as a discovery.")

    source_by_id = {source.source_id: source for source in sources}
    supported_claims = [claim for claim in draft.claims if claim.claim_type == "supported_fact"]
    if not supported_claims:
        raise NoDiscoveryError("The result contained no source-supported factual claim.")
    cited_ids: list[str] = []
    for claim in draft.claims:
        if any(source_id not in source_by_id for source_id in claim.source_ids):
            raise NoDiscoveryError("The synthesis cited a source that was not retrieved.")
        if claim.claim_type == "supported_fact" and not claim.source_ids:
            raise NoDiscoveryError("A factual claim was not linked to retrieved evidence.")
        if claim.claim_type == "supported_fact":
            cited_ids.extend(claim.source_ids)

    evidence = [source_by_id[source_id] for source_id in dict.fromkeys(cited_ids)]
    return ResearchedDiscoveryRead(
        seed_item_id=int(seed["id"]), seed_title=str(seed["title"]), directions=direction_payloads,
        title=draft.title.strip(), discovery=draft.discovery.strip(),
        why_surprising=draft.why_surprising.strip(),
        connection_to_library=(draft.connection_to_library or "").strip() or None,
        what_remains_uncertain=draft.what_remains_uncertain.strip(),
        next_rabbit_hole=draft.next_rabbit_hole.strip(), claims=draft.claims,
        evidence=evidence,
    )
