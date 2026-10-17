"""Bounded, non-persisting research workflow for Discovery Engine v0.1."""

import json
import logging
import os
import re
import uuid
from urllib.parse import urlparse, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from .ai import OpenRouterClient, ResearchedDiscoveryClaim
from .research import ResearchProvider, RetrievedSource


logger = logging.getLogger("app.main")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:authorization|api[_-]?key)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;}]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+"),
    re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}\b"),
)


def _debug_enabled() -> bool:
    return os.getenv("DISCOVERY_DEBUG") == "1"


def _safe_text(value: object, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_url(value: str) -> str:
    """Keep an inspectable source URL without credentials, query parameters, or fragments."""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return _safe_text(urlunsplit((parsed.scheme, host, parsed.path, "", "")), 600)
    except ValueError:
        return "[redacted invalid URL]"


def _safe_source_title(value: str) -> str:
    if value.lstrip().lower().startswith(("http://", "https://")):
        return _safe_url(value)
    return _safe_text(value, 240)


def _debug_record(run_id: str, stage: str, **details: object) -> None:
    if not _debug_enabled():
        return
    logger.info(
        "discovery_debug %s",
        json.dumps({"run_id": run_id, "stage": stage, **details}, ensure_ascii=False, sort_keys=True),
    )


def _no_discovery(run_id: str, reason: str) -> "NoDiscoveryError":
    _debug_record(run_id, "outcome", outcome="no_discovery", reason=_safe_text(reason, 700))
    return NoDiscoveryError(reason)


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
    run_id = uuid.uuid4().hex[:12]
    concepts = seed.get("concepts") if isinstance(seed.get("concepts"), list) else []
    concept_names = [
        _safe_text(concept.get("name"), 160)
        for concept in concepts[:10] if isinstance(concept, dict) and concept.get("name")
    ]
    _debug_record(
        run_id, "seed", title=_safe_text(seed.get("title") or "[untitled]", 240),
        concept_names=concept_names,
    )
    try:
        directions = ai.propose_discovery_directions(seed)
    except Exception as exc:
        _debug_record(
            run_id, "outcome", outcome="error", failed_stage="direction_generation",
            reason=_safe_text(exc, 700),
        )
        raise
    direction_payloads = [direction.model_dump() for direction in directions.directions]
    safe_directions = [{
        "question": _safe_text(direction.get("question"), 400),
        "seed_detail": _safe_text(direction.get("seed_detail"), 300),
        "why_promising": _safe_text(direction.get("why_promising"), 400),
        "discovery_target": _safe_text(direction.get("discovery_target"), 400),
        "search_queries": [_safe_text(query, 300) for query in direction.get("search_queries", [])],
    } for direction in direction_payloads]
    _debug_record(run_id, "directions", directions=safe_directions)
    _debug_record(
        run_id, "research_request",
        submitted_search_queries=[query for direction in safe_directions for query in direction["search_queries"]],
        provider_executed_queries=None,
        query_visibility="OpenRouter's web-search response does not expose the provider-executed queries.",
    )
    try:
        packet = research.research_discovery(seed, direction_payloads)
    except Exception as exc:
        _debug_record(
            run_id, "outcome", outcome="error", failed_stage="research_retrieval",
            reason=_safe_text(exc, 700),
        )
        raise
    sources = _usable_sources(packet.sources)
    _debug_record(run_id, "sources", sources=[{
        "source_id": source.source_id, "title": _safe_source_title(source.title),
        "url": _safe_url(source.url), "excerpt": _safe_text(source.excerpt, 500),
    } for source in sources])
    if not sources:
        raise _no_discovery(run_id, "Research found no inspectable sources strong enough for a discovery.")

    source_payloads = [source.model_dump() for source in sources]
    try:
        draft = ai.synthesize_researched_discovery(seed, direction_payloads, packet.synthesis, source_payloads)
    except Exception as exc:
        _debug_record(
            run_id, "outcome", outcome="error", failed_stage="final_synthesis",
            reason=_safe_text(exc, 700),
        )
        raise
    _debug_record(
        run_id, "synthesis", accepted=draft.accepted,
        rejection_reason=_safe_text(draft.rejection_reason, 700),
        title=_safe_text(draft.title, 240), discovery=_safe_text(draft.discovery, 900),
        why_surprising=_safe_text(draft.why_surprising, 700),
        connection_to_library=_safe_text(draft.connection_to_library, 600) if draft.connection_to_library else None,
        what_remains_uncertain=_safe_text(draft.what_remains_uncertain, 600),
        next_rabbit_hole=_safe_text(draft.next_rabbit_hole, 500),
        claims=[{
            "claim_type": claim.claim_type, "text": _safe_text(claim.text, 500),
            "source_ids": [_safe_text(source_id, 100) for source_id in claim.source_ids],
        } for claim in draft.claims],
        considered_findings=[{
            "direction_index": finding.direction_index, "finding": _safe_text(finding.finding, 600),
            "specificity": _safe_text(finding.specificity, 350),
            "counterintuitiveness": _safe_text(finding.counterintuitiveness, 350),
            "evidence_assessment": _safe_text(finding.evidence_assessment, 400),
            "curiosity_question": _safe_text(finding.curiosity_question, 350),
            "source_ids": [_safe_text(source_id, 100) for source_id in finding.source_ids],
            "selected": finding.selected,
        } for finding in draft.considered_findings],
        selection_rationale=_safe_text(draft.selection_rationale, 700),
    )
    if not draft.accepted:
        raise _no_discovery(
            run_id, draft.rejection_reason or "The retrieved evidence did not support a compelling discovery.",
        )
    required_text = (
        draft.title, draft.discovery, draft.why_surprising,
        draft.what_remains_uncertain, draft.next_rabbit_hole,
    )
    if any(not value.strip() for value in required_text):
        raise _no_discovery(
            run_id, "The researched result was incomplete, so Asterisk did not present it as a discovery.",
        )

    source_by_id = {source.source_id: source for source in sources}
    selected_findings = [finding for finding in draft.considered_findings if finding.selected]
    if len(selected_findings) != 1:
        raise _no_discovery(run_id, "The synthesis did not select exactly one researched finding.")
    if any(finding.direction_index >= len(direction_payloads) for finding in draft.considered_findings):
        raise _no_discovery(run_id, "The synthesis referenced an investigation direction that was not generated.")
    considered_source_ids = {
        source_id for finding in draft.considered_findings for source_id in finding.source_ids
    }
    if any(source_id not in source_by_id for source_id in considered_source_ids):
        raise _no_discovery(run_id, "A considered finding cited a source that was not retrieved.")
    selected_source_ids = set(selected_findings[0].source_ids)
    if not selected_source_ids:
        raise _no_discovery(run_id, "The selected finding had no retrieved evidence.")
    supported_claims = [claim for claim in draft.claims if claim.claim_type == "supported_fact"]
    if not supported_claims:
        raise _no_discovery(run_id, "The result contained no source-supported factual claim.")
    cited_ids: list[str] = []
    for claim in draft.claims:
        if any(source_id not in source_by_id for source_id in claim.source_ids):
            raise _no_discovery(run_id, "The synthesis cited a source that was not retrieved.")
        if claim.claim_type == "supported_fact" and not claim.source_ids:
            raise _no_discovery(run_id, "A factual claim was not linked to retrieved evidence.")
        if claim.claim_type == "supported_fact":
            cited_ids.extend(claim.source_ids)
    if not selected_source_ids.issubset(set(cited_ids)):
        raise _no_discovery(
            run_id, "The final supported facts did not cite the evidence used to select the winning finding.",
        )

    evidence = [source_by_id[source_id] for source_id in dict.fromkeys(cited_ids)]
    result = ResearchedDiscoveryRead(
        seed_item_id=int(seed["id"]), seed_title=str(seed["title"]), directions=direction_payloads,
        title=draft.title.strip(), discovery=draft.discovery.strip(),
        why_surprising=draft.why_surprising.strip(),
        connection_to_library=(draft.connection_to_library or "").strip() or None,
        what_remains_uncertain=draft.what_remains_uncertain.strip(),
        next_rabbit_hole=draft.next_rabbit_hole.strip(), claims=draft.claims,
        evidence=evidence,
    )
    _debug_record(
        run_id, "outcome", outcome="presented", cited_source_ids=[source.source_id for source in evidence],
    )
    return result
