"""Bounded, non-persisting research workflow for Discovery Engine v0.4."""

import json
import hashlib
import logging
import os
import re
import uuid
from urllib.parse import urlparse, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from .ai import ConsideredDiscoveryFinding, OpenRouterClient, ResearchedDiscoveryClaim
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


class DiscoveryCandidateRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    hook: str
    story: str
    twist: str
    library_connection: str | None
    research_direction: str
    originating_curiosity: str
    evidence_assessment: str
    go_deeper: str
    ai_suggested: bool
    evidence: list[DiscoverySourceRead]


class ResearchedDiscoveryRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_item_id: int
    seed_title: str
    directions: list[dict[str, object]]
    discovery_candidates: list[DiscoveryCandidateRead]
    suggested_candidate_id: str | None
    title: str | None
    discovery: str | None
    why_surprising: str | None
    connection_to_library: str | None
    what_remains_uncertain: str | None
    next_rabbit_hole: str | None
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


_FINDING_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "it", "of", "on",
    "that", "the", "their", "then", "to", "was", "were", "with",
}


def _finding_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in _FINDING_STOPWORDS
    }


def _same_finding(first: str, second: str) -> bool:
    normalized_first = " ".join(re.findall(r"[a-z0-9]+", first.casefold()))
    normalized_second = " ".join(re.findall(r"[a-z0-9]+", second.casefold()))
    if normalized_first == normalized_second:
        return True
    first_tokens, second_tokens = _finding_tokens(first), _finding_tokens(second)
    if not first_tokens or not second_tokens:
        return False
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens) >= 0.8


def _eligible_discovery_candidates(
    seed_title: str, directions: list[dict[str, object]], considered_findings: list[ConsideredDiscoveryFinding],
    source_by_id: dict[str, DiscoverySourceRead],
) -> list[DiscoveryCandidateRead]:
    candidates: list[tuple[int, DiscoveryCandidateRead]] = []
    for index, finding in enumerate(considered_findings):
        direction_index = finding.direction_index
        if direction_index >= len(directions) or not finding.source_ids:
            continue
        if any(source_id not in source_by_id for source_id in finding.source_ids):
            continue
        resolved_sources = [source_by_id[source_id] for source_id in dict.fromkeys(finding.source_ids)]
        if not resolved_sources:
            continue
        direction = directions[direction_index]
        identity = "|".join((seed_title.casefold(), str(direction_index), finding.hook.casefold(), finding.story.casefold(), *sorted(finding.source_ids)))
        candidate = DiscoveryCandidateRead(
            candidate_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            hook=finding.hook.strip(), story=finding.story.strip(), twist=finding.twist.strip(),
            library_connection=((finding.library_connection or "").strip() or None),
            research_direction=str(direction.get("question") or "").strip(),
            originating_curiosity=str(direction.get("seed_detail") or seed_title).strip(),
            evidence_assessment=finding.evidence_assessment.strip(),
            go_deeper=finding.go_deeper.strip(), ai_suggested=finding.selected,
            evidence=resolved_sources,
        )
        duplicate_at = next((
            position for position, (_original_index, existing) in enumerate(candidates)
            if _same_finding(f"{existing.hook} {existing.story}", f"{candidate.hook} {candidate.story}")
        ), None)
        if duplicate_at is None:
            candidates.append((index, candidate))
        elif candidate.ai_suggested and not candidates[duplicate_at][1].ai_suggested:
            candidates[duplicate_at] = (index, candidate)

    if len(candidates) > 3:
        suggested = next((entry for entry in candidates if entry[1].ai_suggested), None)
        limited = candidates[:3]
        if suggested is not None and suggested not in limited:
            limited[-1] = suggested
        candidates = sorted(limited, key=lambda entry: entry[0])
    return [candidate for _index, candidate in candidates]


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
            "direction_index": finding.direction_index, "hook": _safe_text(finding.hook, 240),
            "story": _safe_text(finding.story, 800), "twist": _safe_text(finding.twist, 450),
            "library_connection": _safe_text(finding.library_connection, 500) if finding.library_connection else None,
            "evidence_assessment": _safe_text(finding.evidence_assessment, 400),
            "go_deeper": _safe_text(finding.go_deeper, 400),
            "source_ids": [_safe_text(source_id, 100) for source_id in finding.source_ids],
            "selected": finding.selected,
        } for finding in draft.considered_findings],
        selection_rationale=_safe_text(draft.selection_rationale, 700),
    )
    if not draft.accepted:
        raise _no_discovery(
            run_id, draft.rejection_reason or "The retrieved evidence did not support a compelling discovery.",
        )

    source_by_id = {source.source_id: source for source in sources}
    candidates = _eligible_discovery_candidates(
        str(seed["title"]), direction_payloads, draft.considered_findings, source_by_id,
    )
    if not candidates:
        raise _no_discovery(run_id, "No evidence-backed discovery candidates survived source validation.")

    selected_findings = [finding for finding in draft.considered_findings if finding.selected]
    if len(selected_findings) != 1:
        candidates = [candidate.model_copy(update={"ai_suggested": False}) for candidate in candidates]
    suggested_candidate = next((candidate for candidate in candidates if candidate.ai_suggested), None)
    suggested_candidate_id = suggested_candidate.candidate_id if suggested_candidate else None

    full_card_reason: str | None = None
    required_text = (
        draft.title, draft.discovery, draft.why_surprising,
        draft.what_remains_uncertain, draft.next_rabbit_hole,
    )
    if len(selected_findings) != 1:
        full_card_reason = "The synthesis did not select exactly one researched finding."
    elif suggested_candidate is None:
        full_card_reason = "The suggested finding did not survive source validation."
    elif any(not value.strip() for value in required_text):
        full_card_reason = "The selected finding had no complete synthesized explanation."

    cited_ids: list[str] = []
    if full_card_reason is None:
        supported_claims = [claim for claim in draft.claims if claim.claim_type == "supported_fact"]
        if not supported_claims:
            full_card_reason = "The selected explanation contained no source-supported factual claim."
        else:
            for claim in draft.claims:
                if any(source_id not in source_by_id for source_id in claim.source_ids):
                    full_card_reason = "The selected explanation cited a source that was not retrieved."
                    break
                if claim.claim_type == "supported_fact" and not claim.source_ids:
                    full_card_reason = "A factual claim in the selected explanation was not linked to evidence."
                    break
                if claim.claim_type == "supported_fact":
                    cited_ids.extend(claim.source_ids)
            selected_source_ids = {source.source_id for source in suggested_candidate.evidence}
            if full_card_reason is None and not selected_source_ids.issubset(set(cited_ids)):
                full_card_reason = "The selected explanation did not cite the winning finding's evidence."

    evidence = [source_by_id[source_id] for source_id in dict.fromkeys(cited_ids)]
    full_card_available = full_card_reason is None
    result = ResearchedDiscoveryRead(
        seed_item_id=int(seed["id"]), seed_title=str(seed["title"]), directions=direction_payloads,
        discovery_candidates=candidates, suggested_candidate_id=suggested_candidate_id,
        title=draft.title.strip() if full_card_available else None,
        discovery=draft.discovery.strip() if full_card_available else None,
        why_surprising=draft.why_surprising.strip() if full_card_available else None,
        connection_to_library=((draft.connection_to_library or "").strip() or None) if full_card_available else None,
        what_remains_uncertain=draft.what_remains_uncertain.strip() if full_card_available else None,
        next_rabbit_hole=draft.next_rabbit_hole.strip() if full_card_available else None,
        claims=draft.claims if full_card_available else [], evidence=evidence if full_card_available else [],
    )
    _debug_record(
        run_id, "outcome", outcome="presented", candidate_count=len(candidates),
        suggested_candidate_id=suggested_candidate_id, full_card_available=full_card_available,
        full_card_omission_reason=_safe_text(full_card_reason, 500) if full_card_reason else None,
        cited_source_ids=[source.source_id for source in evidence],
    )
    return result
