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

    def extract_concepts(self, item: dict[str, object]) -> ConceptExtraction:
        return self._structured(CONCEPT_SYSTEM_PROMPT, concept_input(item), ConceptExtraction, "concept_extraction", 900)  # type: ignore[return-value]

    def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
        return self._structured(CONNECTION_SYSTEM_PROMPT, connection_input(pairs), CandidateBatch, "sidequest_candidates", 2200)  # type: ignore[return-value]

    def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
        return self._structured(CRITIQUE_SYSTEM_PROMPT, critique_input(candidates), CritiquedConnection, "sidequest_critique", 1400)  # type: ignore[return-value]


def get_ai_client() -> OpenRouterClient:
    return OpenRouterClient()
