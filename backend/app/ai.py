import json
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
    LEADS_SYSTEM_PROMPT,
    UNDERSTANDING_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
    concept_research_input,
    leads_input,
    understanding_input,
    validation_input,
)


class AIConfigurationError(RuntimeError):
    pass


class AIProviderError(RuntimeError):
    pass


class ExtractedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    category: Literal["personal", "general", "mechanism"]
    explanation: str = Field(min_length=1, max_length=600)


class ConceptExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concepts: list[ExtractedConcept] = Field(max_length=6)


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
    events_and_problems: list[str] = Field(max_length=8)
    phenomena_and_processes: list[str] = Field(max_length=8)
    dilemmas_and_decisions: list[str] = Field(max_length=8)
    systems_and_causal_relations: list[str] = Field(max_length=8)


class ConceptualLead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=160)
    proposed_definition: str = Field(min_length=1, max_length=700)
    motivating_detail: str = Field(min_length=1, max_length=900)
    proposed_relationship: str = Field(min_length=1, max_length=900)
    relationship_type: Literal["explicit", "illustrated", "interpretive"]
    research_questions: list[str] = Field(min_length=1, max_length=4)


class ConceptualLeadBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leads: list[ConceptualLead] = Field(min_length=1, max_length=5)


class ValidatedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_name: str = Field(min_length=1, max_length=160)
    accepted: bool
    rejection_reason: str = Field(max_length=700)
    canonical_name: str = Field(max_length=160)
    level: Literal["concept", "mechanism"]
    definition: str = Field(max_length=900)
    domain: str = Field(max_length=120)
    mechanism: str = Field(max_length=1200)
    relationship_type: Literal["explicit", "illustrated", "interpretive"]
    relationship_evidence: str = Field(max_length=1200)
    concept_verification: Literal["verified", "partially_verified", "unverified"]
    relationship_verification: Literal["verified", "supported_interpretation", "unverified"]
    confidence: float = Field(ge=0, le=1)
    concept_source_urls: list[str] = Field(max_length=5)
    relationship_source_urls: list[str] = Field(max_length=5)


class ConceptValidationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[ValidatedConcept] = Field(min_length=1, max_length=7)


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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema.model_json_schema(), "strict": True}},
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

    def research_concepts(self, item: dict[str, object], leads: list[dict[str, object]]) -> tuple[str, list[dict[str, str]]]:
        return self._web_research(CONCEPT_RESEARCH_SYSTEM_PROMPT, concept_research_input(item, leads), "Concept research")

    def _web_research(self, system: str, user: str, label: str) -> tuple[str, list[dict[str, str]]]:
        if not self.api_key:
            raise AIConfigurationError("OPENROUTER_API_KEY is not configured. Add it to your local .env file and restart Asterisk.")
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "tools": [{"type": "openrouter:web_search", "parameters": {"engine": os.getenv("OPENROUTER_SEARCH_ENGINE", "exa"), "max_results": 3, "max_total_results": 12, "max_characters": 2200}}], "max_tokens": 1800}
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

    def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
        return self._structured(CONNECTION_SYSTEM_PROMPT, connection_input(pairs), CandidateBatch, "sidequest_candidates", 2200)  # type: ignore[return-value]

    def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
        return self._structured(CRITIQUE_SYSTEM_PROMPT, critique_input(candidates), CritiquedConnection, "sidequest_critique", 1400)  # type: ignore[return-value]

    def analyze_research(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> EnrichmentAnalysis:
        return self._structured(ENRICHMENT_SYSTEM_PROMPT, enrichment_input(item, synthesis, sources), EnrichmentAnalysis, "research_enrichment", 2200)  # type: ignore[return-value]

    def understand_work(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> WorkUnderstanding:
        return self._structured(UNDERSTANDING_SYSTEM_PROMPT, understanding_input(item, synthesis, sources), WorkUnderstanding, "work_understanding", 1800)  # type: ignore[return-value]

    def generate_conceptual_leads(self, understanding: WorkUnderstanding) -> ConceptualLeadBatch:
        return self._structured(LEADS_SYSTEM_PROMPT, leads_input(understanding.model_dump()), ConceptualLeadBatch, "conceptual_leads", 1800)  # type: ignore[return-value]

    def validate_concepts(self, understanding: WorkUnderstanding, leads: ConceptualLeadBatch, work_sources: list[dict[str, str]], concept_synthesis: str, concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        return self._structured(VALIDATION_SYSTEM_PROMPT, validation_input(understanding.model_dump(), [lead.model_dump() for lead in leads.leads], work_sources, concept_synthesis, concept_sources), ConceptValidationBatch, "concept_validation", 2600)  # type: ignore[return-value]


def get_ai_client() -> OpenRouterClient:
    return OpenRouterClient()
