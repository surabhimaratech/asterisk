import json
import logging
import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .prompts import (
    CONCEPT_SYSTEM_PROMPT,
    CONNECTION_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    concept_input,
    connection_input,
    critique_input,
    ENRICHMENT_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    enrichment_input,
    research_input,
    CONCEPT_RESEARCH_SYSTEM_PROMPT,
    EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
    EVIDENCE_RECOVERY_SYSTEM_PROMPT,
    PROBLEM_SYSTEM_PROMPT,
    RECOVERY_RESEARCH_SYSTEM_PROMPT,
    UNDERSTANDING_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
    evidence_extraction_input,
    evidence_recovery_input,
    problem_input,
    problem_research_input,
    recovery_research_input,
    understanding_input,
    validation_input,
    IDENTITY_SYSTEM_PROMPT,
    identity_input,
)

logger = logging.getLogger(__name__)


class AIConfigurationError(RuntimeError):
    pass


class AIProviderError(RuntimeError):
    pass


def strict_json_schema(schema: dict[str, object]) -> dict[str, object]:
    """Make Pydantic's schema compatible with strict OpenAI-style JSON output."""
    def close_declared_objects(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                close_declared_objects(value)
        elif isinstance(node, list):
            for value in node:
                close_declared_objects(value)

    close_declared_objects(schema)
    return schema


class ExtractedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    category: Literal["personal", "general", "mechanism"]
    explanation: str = Field(min_length=1, max_length=600)


class ConceptExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concepts: list[ExtractedConcept] = Field(max_length=6)


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=300)
    creator: str = Field(max_length=300)
    release_year: int | None = Field(default=None, ge=1000, le=3000)
    media_type: Literal["book", "movie", "tv_series", "podcast", "article"]
    work_scope: Literal["unknown", "individual", "series", "collection", "installment", "adaptation"] = "unknown"
    parent_work_title: str = Field(default="", max_length=300)
    canonical_identity_id: str = Field(default="", max_length=300)
    description: str = Field(min_length=1, max_length=900)
    confidence: float = Field(ge=0, le=1)
    source_ids: list[str] = Field(min_length=1, max_length=5)


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["confirmed", "ambiguous", "unresolved"]
    reason: str = Field(min_length=1, max_length=700)
    candidates: list[EntityCandidate] = Field(max_length=5)


class EnrichedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    level: Literal["topic", "concept", "mechanism"]
    explanation: str = Field(min_length=1, max_length=600)
    relationship: str = Field(min_length=1, max_length=900)
    evidence_status: Literal["supported", "inferred", "unverified"]
    confidence: float = Field(ge=0, le=1)
    source_urls: list[str] = Field(max_length=5)


class EnrichmentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambiguous_identity: bool
    confirmation_question: str = Field(max_length=500)
    resolved_title: str = Field(max_length=300)
    creator: str = Field(max_length=300)
    release_info: str = Field(max_length=200)
    identity_summary: str = Field(max_length=900)
    concepts: list[EnrichedConcept] = Field(max_length=12)


class WorkUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambiguous_identity: bool
    confirmation_question: str = Field(max_length=500)
    resolved_title: str = Field(max_length=300)
    creator: str = Field(max_length=300)
    release_info: str = Field(max_length=200)
    identity_summary: str = Field(max_length=900)
    identity_matches_sources: bool = True
    identity_mismatch_reason: str = Field(default="", max_length=700)
    events_and_problems: list[str] = Field(max_length=8)
    phenomena_and_processes: list[str] = Field(max_length=8)
    dilemmas_and_decisions: list[str] = Field(max_length=8)
    systems_and_causal_relations: list[str] = Field(max_length=8)
    character_relationships_and_tensions: list[str] = Field(default_factory=list, max_length=8)
    worldbuilding_rules_and_constraints: list[str] = Field(default_factory=list, max_length=8)
    narrative_structures_and_techniques: list[str] = Field(default_factory=list, max_length=8)
    tone_humor_and_motifs: list[str] = Field(default_factory=list, max_length=8)
    spoiler_ceiling: Literal["premise_only", "within_progress", "full"] = "premise_only"


class DiscoveryQuality(BaseModel):
    """Discovery value is intentionally separate from evidence verification."""

    model_config = ConfigDict(extra="forbid")
    specificity: Literal["high", "medium", "low"] = "medium"
    intellectual_substance: Literal["high", "medium", "low"] = "medium"
    surprise_potential: Literal["high", "medium", "low"] = "medium"
    rationale: str = Field(default="", max_length=700)


class ResearchProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem_id: str = Field(min_length=1, max_length=60)
    concrete_description: str = Field(min_length=1, max_length=700)
    work_situation: str = Field(min_length=1, max_length=900)
    intellectual_question: str = Field(min_length=1, max_length=500)
    relevant_domains: list[str] = Field(min_length=1, max_length=5)
    evidence_source_ids: list[str] = Field(min_length=1, max_length=5)
    research_queries: list[str] = Field(max_length=4)
    anchor_kind: Literal["event", "relationship", "worldbuilding", "narrative_structure", "tone_or_humor", "philosophical_question", "system", "motif"] = "system"
    concise_label: str = Field(default="", max_length=120)
    interesting_reason: str = Field(default="", max_length=1000)
    distinctiveness: str = Field(default="", max_length=700)
    discovery_mode: Literal["deepen_fascination", "reveal_overlooked_perspective"] = "deepen_fascination"
    external_research_warranted: bool = True
    discovery_quality: DiscoveryQuality = Field(default_factory=DiscoveryQuality)
    spoiler_scope: Literal["premise_only", "within_progress", "full"] = "premise_only"
    mechanism: str = Field(default="", max_length=1000)
    curiosity_invitation: str = Field(default="", max_length=500)
    appeal_dimensions: list[Literal["fascinating", "funny", "beautiful", "strange", "emotionally_resonant", "clever", "unexpected", "comforting", "thought_provoking"]] = Field(default_factory=list, max_length=4)
    pattern_scope: Literal["single_work", "creator_or_series_pattern", "genre_convention", "cross_domain_possibility"] = "single_work"


class ResearchProblemBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problems: list[ResearchProblem] = Field(min_length=1, max_length=8)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=60)
    excerpt: str = Field(min_length=1, max_length=900)
    evidence_form: Literal["quotation", "paraphrase"] = "quotation"


class EvidenceConceptCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=60)
    problem_ids: list[str] = Field(min_length=1, max_length=5)
    discovery_origin: Literal["work_grounded", "research_led"]
    discovery_source_ids: list[str] = Field(min_length=1, max_length=5)
    research_path: str = Field(min_length=1, max_length=900)
    label_kind: Literal["established_term", "descriptive_label"]
    concept_classification: Literal["established_concept", "grounded_interpretation", "speculative_connection"] = "established_concept"
    canonical_name: str = Field(min_length=1, max_length=160)
    aliases: list[str] = Field(max_length=8)
    definition: str = Field(min_length=1, max_length=900)
    domain: str = Field(min_length=1, max_length=120)
    mechanism: str = Field(min_length=1, max_length=1200)
    concept_evidence: list[EvidenceReference] = Field(min_length=1, max_length=5)
    curiosity_hook: str | None = Field(max_length=500)
    uncertainty: str = Field(max_length=700)


class EvidenceConceptBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[EvidenceConceptCandidate] = Field(max_length=8)
    insufficient_evidence_reasons: list[str] = Field(max_length=8)


class EvidenceRecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=60)
    recovered: bool
    reason: str = Field(max_length=700)
    concept_evidence: list[EvidenceReference] = Field(max_length=5)


class ValidatedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=60)
    accepted: bool
    rejection_reason: str = Field(max_length=700)
    discovery_origin: Literal["work_grounded", "research_led"]
    discovery_source_ids: list[str] = Field(max_length=5)
    research_path: str = Field(max_length=900)
    label_kind: Literal["established_term", "descriptive_label"]
    concept_classification: Literal["established_concept", "grounded_interpretation", "speculative_connection"] = "established_concept"
    canonical_name: str = Field(max_length=160)
    aliases: list[str] = Field(max_length=8)
    level: Literal["concept", "mechanism"]
    definition: str = Field(max_length=900)
    concept_evidence: list[EvidenceReference] = Field(max_length=5)
    domain: str = Field(max_length=120)
    mechanism: str = Field(max_length=1200)
    relationship_type: Literal["explicit", "illustrated", "interpretive", "unsupported"]
    relationship_evidence: str = Field(max_length=1200)
    concept_verification: Literal["verified", "partially_verified", "unverified"]
    relationship_verification: Literal["verified", "supported_interpretation", "unverified"]
    confidence: float = Field(ge=0, le=1)
    mechanism_defensibility: Literal["defensible", "uncertain", "unsupported"] = "defensible"
    relationship_source_ids: list[str] = Field(max_length=5)
    curiosity_hook: str | None = Field(max_length=500)


class ConceptValidationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[ValidatedConcept] = Field(max_length=8)


class ConnectionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_a_id: int
    item_b_id: int
    title: str = Field(min_length=1, max_length=180)
    connection: str = Field(min_length=1, max_length=1400)
    bridge: str = Field(min_length=1, max_length=300)
    why_interesting: str = Field(min_length=1, max_length=900)
    where_it_breaks: str = Field(min_length=1, max_length=900)
    new_question: str = Field(min_length=1, max_length=400)


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[ConnectionCandidate] = Field(min_length=1, max_length=6)


class CritiquedConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    rejection_reason: str
    item_a_id: int
    item_b_id: int
    title: str
    connection: str
    bridge: str
    why_interesting: str
    where_it_breaks: str
    new_question: str
    evidence_status: str
    concepts_used: list[str] = Field(max_length=8)
    evidence: list[str] = Field(max_length=8)


class OpenRouterClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30"))

    def _structured(self, system: str, user: str, schema: type[BaseModel], name: str, max_tokens: int) -> BaseModel:
        if not self.api_key:
            raise AIConfigurationError("OPENROUTER_API_KEY is not configured. Add it to your local .env file and restart Asterisk.")
        schema_json = strict_json_schema(schema.model_json_schema())
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema_json, "strict": True}},
            "provider": {"require_parameters": True},
            "max_tokens": max_tokens,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise AIProviderError("The AI request timed out. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise AIProviderError("Could not reach the AI provider.") from exc
        if response.status_code == 429:
            raise AIProviderError("The AI provider rate limit was reached. Please wait and try again.")
        if response.status_code >= 400:
            logger.error("AI provider rejected structured request: status=%s", response.status_code)
            raise AIProviderError(f"The AI provider returned an error ({response.status_code}).")
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            return schema.model_validate(json.loads(text))
        except (KeyError, StopIteration, ValueError, TypeError) as exc:
            raise AIProviderError("The AI provider returned an invalid structured response.") from exc

    def research_item(self, item: dict[str, object]) -> tuple[str, list[dict[str, str]]]:
        if not self.api_key:
            raise AIConfigurationError("OPENROUTER_API_KEY is not configured. Add it to your local .env file and restart Asterisk.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": research_input(item)},
            ],
            "tools": [{"type": "openrouter:web_search", "parameters": {
                "engine": os.getenv("OPENROUTER_SEARCH_ENGINE", "exa"),
                "max_results": 5, "max_total_results": 5, "max_characters": 2500,
            }}],
            "max_tokens": 1300,
        }
        try:
            with httpx.Client(timeout=max(self.timeout, 45)) as client:
                response = client.post(f"{self.base_url}/chat/completions", headers={
                    "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                }, json=payload)
        except httpx.TimeoutException as exc:
            raise AIProviderError("Research timed out. No research was saved; please try again.") from exc
        except httpx.HTTPError as exc:
            raise AIProviderError("Could not reach the research provider.") from exc
        if response.status_code == 429:
            raise AIProviderError("The research provider rate limit was reached. Please wait and try again.")
        if response.status_code >= 400:
            raise AIProviderError(f"The research provider returned an error ({response.status_code}).")
        try:
            message = response.json()["choices"][0]["message"]
            citations: list[dict[str, str]] = []
            for annotation in message.get("annotations", []):
                citation = annotation.get("url_citation", {})
                if citation.get("url"):
                    citations.append({"url": citation["url"], "title": citation.get("title") or citation["url"], "excerpt": citation.get("content") or ""})
            if not citations:
                raise AIProviderError("Research returned no inspectable sources. Nothing was saved.")
            return message["content"], citations
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AIProviderError):
                raise
            raise AIProviderError("The research provider returned an invalid response.") from exc

    def research_problems(self, _item: dict[str, object], problems: list[dict[str, object]]) -> tuple[str, list[dict[str, str]]]:
        return self._web_research(CONCEPT_RESEARCH_SYSTEM_PROMPT, problem_research_input(problems), "Concept research")

    def research_candidate_evidence(self, candidate: dict[str, object]) -> tuple[str, list[dict[str, str]]]:
        max_results = max(1, min(int(os.getenv("EVIDENCE_RECOVERY_MAX_SEARCH_RESULTS", "4")), 5))
        return self._web_research(RECOVERY_RESEARCH_SYSTEM_PROMPT, recovery_research_input(candidate), "Evidence recovery", max_results=2, max_total_results=max_results)

    def _web_research(self, system: str, user: str, label: str, max_results: int = 3, max_total_results: int = 12) -> tuple[str, list[dict[str, str]]]:
        if not self.api_key:
            raise AIConfigurationError("OPENROUTER_API_KEY is not configured. Add it to your local .env file and restart Asterisk.")
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "tools": [{"type": "openrouter:web_search", "parameters": {"engine": os.getenv("OPENROUTER_SEARCH_ENGINE", "exa"), "max_results": max_results, "max_total_results": max_total_results, "max_characters": 2200}}], "max_tokens": 1800}
        try:
            with httpx.Client(timeout=max(self.timeout, 45)) as client:
                response = client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload)
        except httpx.TimeoutException as exc:
            raise AIProviderError(f"{label} timed out. Existing research was preserved.") from exc
        except httpx.HTTPError as exc:
            raise AIProviderError(f"Could not reach the provider during {label.lower()}.") from exc
        if response.status_code == 429:
            raise AIProviderError("The research provider rate limit was reached. Please wait and try again.")
        if response.status_code >= 400:
            raise AIProviderError(f"The research provider returned an error ({response.status_code}).")
        try:
            message = response.json()["choices"][0]["message"]
            citations = []
            for annotation in message.get("annotations", []):
                citation = annotation.get("url_citation", {})
                if citation.get("url"):
                    citations.append({"url": citation["url"], "title": citation.get("title") or citation["url"], "excerpt": citation.get("content") or ""})
            if not citations:
                raise AIProviderError(f"{label} returned no inspectable sources. Existing research was preserved.")
            return message["content"], citations
        except (KeyError, TypeError, ValueError) as exc:
            raise AIProviderError(f"{label} returned an invalid response.") from exc

    def extract_concepts(self, item: dict[str, object]) -> ConceptExtraction:
        return self._structured(CONCEPT_SYSTEM_PROMPT, concept_input(item), ConceptExtraction, "concept_extraction", 900)  # type: ignore[return-value]

    def resolve_identity(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
        return self._structured(IDENTITY_SYSTEM_PROMPT, identity_input(item, synthesis, sources), EntityResolution, "entity_resolution", 1400)  # type: ignore[return-value]

    def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
        return self._structured(CONNECTION_SYSTEM_PROMPT, connection_input(pairs), CandidateBatch, "sidequest_candidates", 2200)  # type: ignore[return-value]

    def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
        return self._structured(CRITIQUE_SYSTEM_PROMPT, critique_input(candidates), CritiquedConnection, "sidequest_critique", 1400)  # type: ignore[return-value]

    def analyze_research(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> EnrichmentAnalysis:
        return self._structured(ENRICHMENT_SYSTEM_PROMPT, enrichment_input(item, synthesis, sources), EnrichmentAnalysis, "research_enrichment", 2200)  # type: ignore[return-value]

    def understand_work(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> WorkUnderstanding:
        return self._structured(UNDERSTANDING_SYSTEM_PROMPT, understanding_input(item, synthesis, sources), WorkUnderstanding, "work_understanding", 1800)  # type: ignore[return-value]

    def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]], curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
        return self._structured(PROBLEM_SYSTEM_PROMPT, problem_input(understanding.model_dump(), work_sources, curiosity_profile), ResearchProblemBatch, "research_problems", 2200)  # type: ignore[return-value]

    def extract_evidence_concepts(self, problems: ResearchProblemBatch, concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        return self._structured(EVIDENCE_EXTRACTION_SYSTEM_PROMPT, evidence_extraction_input([problem.model_dump() for problem in problems.problems], concept_synthesis, concept_sources), EvidenceConceptBatch, "evidence_concepts", 2400)  # type: ignore[return-value]

    def recover_candidate_evidence(self, candidate: EvidenceConceptCandidate, recovery_synthesis: str, recovery_sources: list[dict[str, str]]) -> EvidenceRecoveryResult:
        return self._structured(EVIDENCE_RECOVERY_SYSTEM_PROMPT, evidence_recovery_input(candidate.model_dump(), recovery_synthesis, recovery_sources), EvidenceRecoveryResult, "evidence_recovery", 1000)  # type: ignore[return-value]

    def validate_concepts(self, understanding: WorkUnderstanding, problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, work_sources: list[dict[str, str]], concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        return self._structured(VALIDATION_SYSTEM_PROMPT, validation_input(understanding.model_dump(), [problem.model_dump() for problem in problems.problems], [candidate.model_dump() for candidate in candidates.candidates], work_sources, concept_sources), ConceptValidationBatch, "concept_validation", 3000)  # type: ignore[return-value]


def get_ai_client() -> OpenRouterClient:
    return OpenRouterClient()
