import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import re
import shutil
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, delete, inspect, select, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .ai import AIConfigurationError, AIProviderError, ConceptValidationBatch, EvidenceConceptBatch, EvidenceConceptCandidate, EntityCandidate, OpenRouterClient, ResearchProblem, ResearchProblemBatch, get_ai_client
from .quality import candidate_evidence_rejection, concept_quality_rejection, deterministic_rejection_reason, rebind_exact_evidence, research_problem_rejection
from .research import ResearchProvider, get_research_provider

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "asterisk.db"
LEGACY_DB_PATH = BACKEND_DIR / "curiosity.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
configured_database_url = os.getenv("DATABASE_URL")
if configured_database_url is None and not DB_PATH.exists() and LEGACY_DB_PATH.exists():
    shutil.copy2(LEGACY_DB_PATH, DB_PATH)

DATABASE_URL = configured_database_url or f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

class Kind(str, Enum):
    book = "book"
    movie = "movie"
    tv_series = "tv_series"
    podcast = "podcast"
    article = "article"
    concept = "concept"
    question = "question"
    thought = "thought"


class WorkScope(str, Enum):
    unknown = "unknown"
    individual = "individual"
    series = "series"
    collection = "collection"
    installment = "installment"
    adaptation = "adaptation"


class ProgressStatus(str, Enum):
    unknown = "unknown"
    planned = "planned"
    in_progress = "in_progress"
    paused = "paused"
    finished = "finished"
    abandoned = "abandoned"
    rereading = "rereading"
    rewatching = "rewatching"


class ProgressUnit(str, Enum):
    installment = "installment"
    book = "book"
    season = "season"
    episode = "episode"
    percent = "percent"


class SpoilerPolicy(str, Enum):
    avoid = "avoid"
    through_progress = "through_progress"
    full = "full"

class CuriosityItem(Base):
    __tablename__ = "curiosity_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(240), index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    creator: Mapped[str | None] = mapped_column(String(300), nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    work_scope: Mapped[str] = mapped_column(String(30), default=WorkScope.unknown.value)
    parent_work_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    canonical_identity_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    progress_status: Mapped[str] = mapped_column(String(30), default=ProgressStatus.unknown.value)
    progress_current: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_unit: Mapped[str] = mapped_column(String(30), default=ProgressUnit.installment.value)
    spoiler_policy: Mapped[str] = mapped_column(String(30), default=SpoilerPolicy.avoid.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class CanonicalConcept(Base):
    __tablename__ = "canonical_concepts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    normalized_name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    definition: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(120))
    mechanism: Mapped[str] = mapped_column(Text)
    verification_status: Mapped[str] = mapped_column(String(30))
    aliases_json: Mapped[str] = mapped_column(Text, default="[]")
    label_kind: Mapped[str] = mapped_column(String(30), default="established_term")
    provenance: Mapped[str] = mapped_column(String(20), default="ai")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemConcept(Base):
    __tablename__ = "item_concepts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    canonical_concept_id: Mapped[int | None] = mapped_column(ForeignKey("canonical_concepts.id"), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(20))
    origin: Mapped[str] = mapped_column(String(20))
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relationship: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    canonical_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(120), nullable=True)
    mechanism: Mapped[str | None] = mapped_column(Text, nullable=True)
    relationship_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relationship_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    concept_verification: Mapped[str | None] = mapped_column(String(30), nullable=True)
    relationship_verification: Mapped[str | None] = mapped_column(String(30), nullable=True)
    label_kind: Mapped[str | None] = mapped_column(String(30), nullable=True)
    curiosity_hook: Mapped[str | None] = mapped_column(Text, nullable=True)
    discovery_origin: Mapped[str | None] = mapped_column(String(30), nullable=True)
    research_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    discovery_source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    initial_validation_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovery_attempt_json: Mapped[str] = mapped_column(Text, default="{}")
    final_validation_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_audit_id: Mapped[int | None] = mapped_column(ForeignKey("concept_lead_audits.id"), index=True, nullable=True)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    concept_classification: Mapped[str | None] = mapped_column(String(40), nullable=True)
    evidence_form: Mapped[str | None] = mapped_column(String(20), nullable=True)
    feedback_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    discovery_mode: Mapped[str | None] = mapped_column(String(40), nullable=True)
    discovery_quality_json: Mapped[str] = mapped_column(Text, default="{}")
    appeal_dimensions_json: Mapped[str] = mapped_column(Text, default="[]")
    pattern_scope: Mapped[str | None] = mapped_column(String(40), nullable=True)
    spoiler_scope: Mapped[str] = mapped_column(String(30), default="premise_only")
    interested: Mapped[bool] = mapped_column(default=False)
    interest_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    interested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    loved: Mapped[bool] = mapped_column(default=False)
    love_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    loved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    surprised: Mapped[bool] = mapped_column(default=False)
    surprise_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    surprised_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ExplorationInvitation(Base):
    __tablename__ = "exploration_invitations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    source_concept_id: Mapped[int | None] = mapped_column(ForeignKey("item_concepts.id"), index=True, nullable=True)
    source_feedback_key: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    source_concept_name: Mapped[str] = mapped_column(String(160))
    question: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Sidequest(Base):
    __tablename__ = "sidequests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_a_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    item_b_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    title: Mapped[str] = mapped_column(String(180))
    connection: Mapped[str] = mapped_column(Text)
    bridge: Mapped[str] = mapped_column(String(300))
    why_interesting: Mapped[str] = mapped_column(Text)
    where_it_breaks: Mapped[str] = mapped_column(Text)
    new_question: Mapped[str] = mapped_column(String(400))
    evidence_status: Mapped[str] = mapped_column(String(240))
    concepts_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    origin_item_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    discovery_kind: Mapped[str] = mapped_column(String(40), default="unexpected_connection")
    invitation: Mapped[str] = mapped_column(String(300), default="")
    dedupe_key: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemEnrichment(Base):
    __tablename__ = "item_enrichments"
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(30))
    resolved_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    creator: Mapped[str | None] = mapped_column(String(300), nullable=True)
    release_info: Mapped[str | None] = mapped_column(String(200), nullable=True)
    identity_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmation_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    identity_status: Mapped[str] = mapped_column(String(30), default="unresolved")
    identity_key: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_media_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    resolved_work_scope: Mapped[str | None] = mapped_column(String(30), nullable=True)
    parent_work_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    canonical_identity_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    identity_candidates_json: Mapped[str] = mapped_column(Text, default="[]")
    identity_source_snapshot_json: Mapped[str] = mapped_column(Text, default="[]")
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    researched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ResearchSource(Base):
    __tablename__ = "research_sources"
    __table_args__ = (UniqueConstraint("item_id", "url", name="uq_research_source_item_url"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    url: Mapped[str] = mapped_column(String(2000))
    title: Mapped[str] = mapped_column(String(500))
    publisher: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_type: Mapped[str] = mapped_column(String(40), default="web")
    evidence_roles_json: Mapped[str] = mapped_column(Text, default="[]")
    excerpt: Mapped[str] = mapped_column(Text)
    retrieval_status: Mapped[str] = mapped_column(String(30), default="retrieved")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class ConceptSource(Base):
    __tablename__ = "concept_sources"
    __table_args__ = (UniqueConstraint("concept_id", "source_id", name="uq_concept_source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("item_concepts.id"), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("research_sources.id"), index=True)
    support_kind: Mapped[str] = mapped_column(String(50), default="relationship")

class ConceptLeadAudit(Base):
    __tablename__ = "concept_lead_audits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    proposed_definition: Mapped[str] = mapped_column(Text)
    motivating_detail: Mapped[str] = mapped_column(Text)
    proposed_relationship: Mapped[str] = mapped_column(Text)
    relationship_type: Mapped[str] = mapped_column(String(20))
    research_questions_json: Mapped[str] = mapped_column(Text, default="[]")
    accepted: Mapped[bool] = mapped_column(default=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage: Mapped[str] = mapped_column(String(30), default="concept_candidate")
    discovery_origin: Mapped[str | None] = mapped_column(String(30), nullable=True)
    discovery_source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    research_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    initial_validation_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovery_attempt_json: Mapped[str] = mapped_column(Text, default="{}")
    final_validation_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_snapshot_json: Mapped[str] = mapped_column(Text, default="{}")
    validation_snapshot_json: Mapped[str] = mapped_column(Text, default="{}")
    source_snapshot_json: Mapped[str] = mapped_column(Text, default="[]")
    automated_evidence_status: Mapped[str] = mapped_column(String(30), default="unverified")
    human_review_status: Mapped[str] = mapped_column(String(20), default="pending")
    human_review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_history_json: Mapped[str] = mapped_column(Text, default="[]")
    kept_concept_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evidence_review_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    concept_classification: Mapped[str | None] = mapped_column(String(40), nullable=True)
    evidence_form: Mapped[str | None] = mapped_column(String(20), nullable=True)
    feedback_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    discovery_mode: Mapped[str | None] = mapped_column(String(40), nullable=True)
    discovery_quality_json: Mapped[str] = mapped_column(Text, default="{}")
    interested: Mapped[bool] = mapped_column(default=False)
    interest_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    interested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    loved: Mapped[bool] = mapped_column(default=False)
    love_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    loved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    surprised: Mapped[bool] = mapped_column(default=False)
    surprise_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    surprised_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemCreate(BaseModel):
    kind: Kind
    title: str = Field(min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)
    creator: str | None = Field(default=None, max_length=300)
    release_year: int | None = Field(default=None, ge=1000, le=3000)
    work_scope: WorkScope = WorkScope.unknown
    parent_work_title: str | None = Field(default=None, max_length=300)
    canonical_identity_id: str | None = Field(default=None, max_length=300)
    progress_status: ProgressStatus = ProgressStatus.unknown
    progress_current: int | None = Field(default=None, ge=0, le=100000)
    progress_total: int | None = Field(default=None, ge=1, le=100000)
    progress_unit: ProgressUnit = ProgressUnit.installment
    spoiler_policy: SpoilerPolicy = SpoilerPolicy.avoid

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        return normalize_url(value)

class ItemUpdate(BaseModel):
    kind: Kind | None = None
    title: str | None = Field(default=None, min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)
    creator: str | None = Field(default=None, max_length=300)
    release_year: int | None = Field(default=None, ge=1000, le=3000)
    work_scope: WorkScope | None = None
    parent_work_title: str | None = Field(default=None, max_length=300)
    canonical_identity_id: str | None = Field(default=None, max_length=300)
    progress_status: ProgressStatus | None = None
    progress_current: int | None = Field(default=None, ge=0, le=100000)
    progress_total: int | None = Field(default=None, ge=1, le=100000)
    progress_unit: ProgressUnit | None = None
    spoiler_policy: SpoilerPolicy | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        return normalize_url(value)

class ItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: Kind
    title: str
    note: str | None
    url: str | None
    creator: str | None
    release_year: int | None
    work_scope: WorkScope
    parent_work_title: str | None
    canonical_identity_id: str | None
    progress_status: ProgressStatus
    progress_current: int | None
    progress_total: int | None
    progress_unit: ProgressUnit
    spoiler_policy: SpoilerPolicy
    created_at: datetime

class ConceptCategory(str, Enum):
    personal = "personal"
    general = "general"
    mechanism = "mechanism"

class ConceptOrigin(str, Enum):
    ai = "ai"
    user = "user"
    user_kept = "user_kept"

class ConceptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: ConceptCategory
    explanation: str | None = Field(default=None, max_length=600)

class ConceptRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    item_id: int
    canonical_concept_id: int | None
    name: str
    category: ConceptCategory
    origin: ConceptOrigin
    explanation: str | None
    level: str | None
    relationship: str | None
    evidence_status: str | None
    confidence: float | None
    source_ids: list[int] = Field(default_factory=list)
    source_support: list[dict[str, int | str]] = Field(default_factory=list)
    canonical_name: str | None
    definition: str | None
    domain: str | None
    mechanism: str | None
    relationship_type: str | None
    relationship_evidence: str | None
    concept_verification: str | None
    relationship_verification: str | None
    label_kind: str | None
    curiosity_hook: str | None
    discovery_origin: str | None
    research_path: str | None
    discovery_source_ids: list[str] = Field(default_factory=list)
    initial_validation_outcome: str | None
    recovery_attempt: dict[str, object] = Field(default_factory=dict)
    final_validation_outcome: str | None
    review_audit_id: int | None
    user_note: str | None
    concept_classification: str | None
    evidence_form: str | None
    feedback_key: str | None
    discovery_mode: str | None
    discovery_quality: dict[str, object] = Field(default_factory=dict)
    appeal_dimensions: list[str] = Field(default_factory=list)
    pattern_scope: str | None
    spoiler_scope: str
    interested: bool
    interest_note: str | None
    interested_at: datetime | None
    loved: bool
    love_note: str | None
    loved_at: datetime | None
    surprised: bool
    surprise_note: str | None
    surprised_at: datetime | None
    created_at: datetime

class ReviewDecision(BaseModel):
    decision: Literal["keep", "dismiss", "pending"]
    note: str | None = Field(default=None, max_length=2000)

class RejectedFindingRead(BaseModel):
    id: int
    item_id: int
    item_title: str
    item_kind: Kind
    name: str
    stage: str
    proposed_definition: str
    motivating_detail: str
    proposed_relationship: str
    relationship_type: str
    rejection_reason: str
    discovery_origin: str | None
    research_path: str | None
    research_questions: list[str]
    initial_validation_outcome: str | None
    recovery_attempt: dict[str, object]
    final_validation_outcome: str | None
    automated_evidence_status: str
    human_review_status: str
    human_review_note: str | None
    review_history: list[dict[str, object]]
    kept_concept_id: int | None
    evidence_review_detail: str | None
    last_evaluated_at: datetime | None
    candidate: dict[str, object]
    validation: dict[str, object]
    sources: list[dict[str, object]]
    superseded_at: datetime | None
    concept_classification: str | None
    evidence_form: str | None
    feedback_key: str | None
    discovery_mode: str | None
    discovery_quality: dict[str, object] = Field(default_factory=dict)
    interested: bool
    interest_note: str | None
    interested_at: datetime | None
    loved: bool
    love_note: str | None
    loved_at: datetime | None
    surprised: bool
    surprise_note: str | None
    surprised_at: datetime | None
    created_at: datetime

class SidequestRead(BaseModel):
    id: int
    item_a_id: int
    item_b_id: int
    item_a_title: str
    item_b_title: str
    title: str
    connection: str
    bridge: str
    why_interesting: str
    where_it_breaks: str
    new_question: str
    evidence_status: str
    concepts_used: list[str]
    evidence: list[str]
    origin_item_ids: list[int]
    origin_titles: list[str]
    discovery_kind: str
    invitation: str
    created_at: datetime


class ExplorationInvitationRead(BaseModel):
    id: int
    item_id: int
    item_title: str
    source_concept_id: int | None
    active_concept_id: int | None
    source_concept_name: str
    question: str
    status: Literal["queued", "dismissed"]
    user_note: str | None
    available: bool
    created_at: datetime
    updated_at: datetime


class ExplorationInvitationUpdate(BaseModel):
    status: Literal["queued", "dismissed"]
    note: str | None = Field(default=None, max_length=2000)

class ResearchSourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    url: str
    title: str
    publisher: str | None
    source_type: str
    evidence_roles: list[str] = Field(default_factory=list)
    excerpt: str
    retrieval_status: str
    retrieved_at: datetime

class EnrichmentRead(BaseModel):
    status: str
    resolved_title: str | None
    creator: str | None
    release_info: str | None
    identity_summary: str | None
    confirmation_question: str | None
    identity_status: str
    identity_key: str | None
    resolved_media_type: str | None
    resolved_work_scope: str | None
    parent_work_title: str | None
    canonical_identity_id: str | None
    identity_confidence: float | None
    identity_candidates: list[dict[str, object]] = Field(default_factory=list)
    identity_sources: list[dict[str, object]] = Field(default_factory=list)
    confirmed_at: datetime | None
    researched_at: datetime | None
    sources: list[ResearchSourceRead]
    concepts: list[ConceptRead]
    rejected_leads: list[RejectedFindingRead] = Field(default_factory=list)

class EnrichmentRequest(BaseModel):
    identity_hint: str | None = Field(default=None, max_length=500)

class IdentityConfirmation(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=80)

class InterestUpdate(BaseModel):
    interested: bool
    note: str | None = Field(default=None, max_length=2000)


class DiscoveryFeedbackUpdate(BaseModel):
    signal: Literal["love", "surprise"]
    active: bool
    note: str | None = Field(default=None, max_length=2000)


class CuriosityProfileRead(BaseModel):
    hypotheses: list[dict[str, object]] = Field(default_factory=list)
    authored_lenses: list[str] = Field(default_factory=list)
    exploration_budget: int = 1

def normalize_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return str(TypeAdapter(AnyHttpUrl).validate_python(value.strip()))

def get_db():
    with SessionLocal() as session:
        yield session

Base.metadata.create_all(bind=engine)

def migrate_additive_columns(target_engine=engine, database_url: str = DATABASE_URL) -> None:
    """Small additive migration runner for existing local SQLite databases."""
    if not database_url.startswith("sqlite"):
        return
    ExplorationInvitation.__table__.create(bind=target_engine, checkfirst=True)
    additions = {
        "curiosity_items": {"creator": "VARCHAR(300)", "release_year": "INTEGER", "work_scope": "VARCHAR(30) NOT NULL DEFAULT 'unknown'", "parent_work_title": "VARCHAR(300)", "canonical_identity_id": "VARCHAR(300)", "progress_status": "VARCHAR(30) NOT NULL DEFAULT 'unknown'", "progress_current": "INTEGER", "progress_total": "INTEGER", "progress_unit": "VARCHAR(30) NOT NULL DEFAULT 'installment'", "spoiler_policy": "VARCHAR(30) NOT NULL DEFAULT 'avoid'"},
        "item_concepts": {"canonical_concept_id": "INTEGER REFERENCES canonical_concepts(id)", "level": "VARCHAR(20)", "relationship": "TEXT", "evidence_status": "VARCHAR(40)", "confidence": "FLOAT", "canonical_name": "VARCHAR(160)", "definition": "TEXT", "domain": "VARCHAR(120)", "mechanism": "TEXT", "relationship_type": "VARCHAR(20)", "relationship_evidence": "TEXT", "concept_verification": "VARCHAR(30)", "label_kind": "VARCHAR(30)", "curiosity_hook": "TEXT", "discovery_origin": "VARCHAR(30)", "research_path": "TEXT", "discovery_source_ids_json": "TEXT NOT NULL DEFAULT '[]'", "initial_validation_outcome": "TEXT", "recovery_attempt_json": "TEXT NOT NULL DEFAULT '{}'", "final_validation_outcome": "TEXT", "review_audit_id": "INTEGER REFERENCES concept_lead_audits(id)", "user_note": "TEXT", "concept_classification": "VARCHAR(40)", "evidence_form": "VARCHAR(20)", "feedback_key": "VARCHAR(64)", "discovery_mode": "VARCHAR(40)", "discovery_quality_json": "TEXT NOT NULL DEFAULT '{}'", "appeal_dimensions_json": "TEXT NOT NULL DEFAULT '[]'", "pattern_scope": "VARCHAR(40)", "spoiler_scope": "VARCHAR(30) NOT NULL DEFAULT 'premise_only'", "interested": "BOOLEAN NOT NULL DEFAULT 0", "interest_note": "TEXT", "interested_at": "DATETIME", "loved": "BOOLEAN NOT NULL DEFAULT 0", "love_note": "TEXT", "loved_at": "DATETIME", "surprised": "BOOLEAN NOT NULL DEFAULT 0", "surprise_note": "TEXT", "surprised_at": "DATETIME", "superseded_at": "DATETIME"},
        "sidequests": {"concepts_json": "TEXT NOT NULL DEFAULT '[]'", "evidence_json": "TEXT NOT NULL DEFAULT '[]'", "origin_item_ids_json": "TEXT NOT NULL DEFAULT '[]'", "discovery_kind": "VARCHAR(40) NOT NULL DEFAULT 'unexpected_connection'", "invitation": "VARCHAR(300) NOT NULL DEFAULT ''", "dedupe_key": "VARCHAR(64)"},
        "concept_sources": {"support_kind": "VARCHAR(20) NOT NULL DEFAULT 'relationship'"},
        "canonical_concepts": {"aliases_json": "TEXT NOT NULL DEFAULT '[]'", "label_kind": "VARCHAR(30) NOT NULL DEFAULT 'established_term'"},
        "concept_lead_audits": {"stage": "VARCHAR(30) NOT NULL DEFAULT 'concept_candidate'", "discovery_origin": "VARCHAR(30)", "discovery_source_ids_json": "TEXT NOT NULL DEFAULT '[]'", "research_path": "TEXT", "initial_validation_outcome": "TEXT", "recovery_attempt_json": "TEXT NOT NULL DEFAULT '{}'", "final_validation_outcome": "TEXT", "candidate_snapshot_json": "TEXT NOT NULL DEFAULT '{}'", "validation_snapshot_json": "TEXT NOT NULL DEFAULT '{}'", "source_snapshot_json": "TEXT NOT NULL DEFAULT '[]'", "automated_evidence_status": "VARCHAR(30) NOT NULL DEFAULT 'unverified'", "human_review_status": "VARCHAR(20) NOT NULL DEFAULT 'pending'", "human_review_note": "TEXT", "review_history_json": "TEXT NOT NULL DEFAULT '[]'", "kept_concept_id": "INTEGER", "last_evaluated_at": "DATETIME", "evidence_review_detail": "TEXT", "superseded_at": "DATETIME", "concept_classification": "VARCHAR(40)", "evidence_form": "VARCHAR(20)", "feedback_key": "VARCHAR(64)", "discovery_mode": "VARCHAR(40)", "discovery_quality_json": "TEXT NOT NULL DEFAULT '{}'", "interested": "BOOLEAN NOT NULL DEFAULT 0", "interest_note": "TEXT", "interested_at": "DATETIME", "loved": "BOOLEAN NOT NULL DEFAULT 0", "love_note": "TEXT", "loved_at": "DATETIME", "surprised": "BOOLEAN NOT NULL DEFAULT 0", "surprise_note": "TEXT", "surprised_at": "DATETIME"},
        "item_enrichments": {"identity_status": "VARCHAR(30) NOT NULL DEFAULT 'unresolved'", "identity_key": "VARCHAR(300)", "resolved_media_type": "VARCHAR(40)", "resolved_work_scope": "VARCHAR(30)", "parent_work_title": "VARCHAR(300)", "canonical_identity_id": "VARCHAR(300)", "identity_confidence": "FLOAT", "identity_candidates_json": "TEXT NOT NULL DEFAULT '[]'", "identity_source_snapshot_json": "TEXT NOT NULL DEFAULT '[]'", "confirmed_at": "DATETIME"},
        "research_sources": {"evidence_roles_json": "TEXT NOT NULL DEFAULT '[]'", "superseded_at": "DATETIME"},
    }
    with target_engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version VARCHAR(120) PRIMARY KEY, applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        schema = inspect(connection)
        for table, columns in additions.items():
            existing = {column["name"] for column in schema.get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        migration_version = "2026-10-13-known-tv-series"
        already_applied = connection.exec_driver_sql(
            "SELECT 1 FROM schema_migrations WHERE version = ?", (migration_version,)
        ).fetchone()
        if already_applied is None:
            # One-time correction for the three legacy records known to have been
            # entered before TV Series was a distinct type. Future same-titled
            # movie entries must remain exactly as the user categorized them.
            connection.exec_driver_sql(
                "UPDATE curiosity_items SET kind = 'tv_series' "
                "WHERE kind = 'movie' AND lower(trim(title)) IN ('castle', 'game of thrones', 'normal people')"
            )
            known_tv = "SELECT id FROM curiosity_items WHERE kind = 'tv_series' AND lower(trim(title)) IN ('castle', 'game of thrones', 'normal people')"
            connection.exec_driver_sql(f"UPDATE item_concepts SET superseded_at = COALESCE(superseded_at, CURRENT_TIMESTAMP) WHERE origin IN ('ai', 'user_kept') AND item_id IN ({known_tv}) AND item_id IN (SELECT item_id FROM item_enrichments WHERE identity_status != 'confirmed')")
            connection.exec_driver_sql(f"UPDATE research_sources SET superseded_at = COALESCE(superseded_at, CURRENT_TIMESTAMP) WHERE item_id IN ({known_tv}) AND item_id IN (SELECT item_id FROM item_enrichments WHERE identity_status != 'confirmed')")
            connection.exec_driver_sql(f"UPDATE concept_lead_audits SET superseded_at = COALESCE(superseded_at, CURRENT_TIMESTAMP) WHERE item_id IN ({known_tv}) AND item_id IN (SELECT item_id FROM item_enrichments WHERE identity_status != 'confirmed')")
            connection.exec_driver_sql(f"UPDATE item_enrichments SET status = 'needs_confirmation' WHERE item_id IN ({known_tv}) AND identity_status != 'confirmed'")
            connection.exec_driver_sql("INSERT INTO schema_migrations (version) VALUES (?)", (migration_version,))

migrate_additive_columns()
app = FastAPI(title="Asterisk", description="There's more to the story.", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def ai_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AIConfigurationError):
        return HTTPException(503, str(exc))
    return HTTPException(502, str(exc))

def sidequest_rejection_detail(reason: str | None = None) -> str:
    guidance = "Try adding more specific notes or concepts to each curiosity, then try again."
    if reason and reason.strip():
        return f"Asterisk found possible links, but none were strong enough to save as a Sidequest. {reason.strip()} {guidance}"
    return f"Asterisk did not find a connection strong enough to save as a Sidequest. {guidance}"

def sidequest_read(sidequest: Sidequest, db: Session) -> SidequestRead:
    item_a = db.get(CuriosityItem, sidequest.item_a_id)
    item_b = db.get(CuriosityItem, sidequest.item_b_id)
    origin_ids = json.loads(sidequest.origin_item_ids_json or "[]") or list(dict.fromkeys((sidequest.item_a_id, sidequest.item_b_id)))
    origin_items = [db.get(CuriosityItem, item_id) for item_id in origin_ids]
    return SidequestRead(
        **{column.name: getattr(sidequest, column.name) for column in Sidequest.__table__.columns},
        item_a_title=item_a.title if item_a else "Deleted curiosity",
        item_b_title=item_b.title if item_b else "Deleted curiosity",
        concepts_used=json.loads(sidequest.concepts_json or "[]"),
        evidence=json.loads(sidequest.evidence_json or "[]"),
        origin_item_ids=origin_ids,
        origin_titles=[item.title if item else "Deleted curiosity" for item in origin_items],
    )


def active_invitation_concept(invitation: ExplorationInvitation, db: Session) -> ItemConcept | None:
    concept = db.get(ItemConcept, invitation.source_concept_id) if invitation.source_concept_id is not None else None
    if concept is not None and concept.superseded_at is None and concept.item_id == invitation.item_id:
        return concept
    if invitation.source_feedback_key:
        return db.scalar(select(ItemConcept).where(
            ItemConcept.item_id == invitation.item_id,
            ItemConcept.feedback_key == invitation.source_feedback_key,
            ItemConcept.superseded_at.is_(None),
        ).order_by(ItemConcept.created_at.desc(), ItemConcept.id.desc()))
    return None


def exploration_invitation_read(invitation: ExplorationInvitation, db: Session) -> ExplorationInvitationRead:
    item = db.get(CuriosityItem, invitation.item_id)
    active_concept = active_invitation_concept(invitation, db)
    return ExplorationInvitationRead(
        id=invitation.id, item_id=invitation.item_id,
        item_title=item.title if item else "Deleted curiosity",
        source_concept_id=invitation.source_concept_id,
        active_concept_id=active_concept.id if active_concept else None,
        source_concept_name=invitation.source_concept_name,
        question=invitation.question, status=invitation.status,
        user_note=invitation.user_note, available=active_concept is not None,
        created_at=invitation.created_at, updated_at=invitation.updated_at,
    )


def concept_read(concept: ItemConcept, db: Session) -> ConceptRead:
    source_links = list(db.scalars(select(ConceptSource).where(ConceptSource.concept_id == concept.id).order_by(ConceptSource.id)).all())
    source_ids = [link.source_id for link in source_links]
    return ConceptRead(
        id=concept.id, item_id=concept.item_id, canonical_concept_id=concept.canonical_concept_id, name=concept.name, category=concept.category,
        origin=concept.origin, explanation=concept.explanation, level=concept.level,
        relationship=concept.relationship, evidence_status=concept.evidence_status,
        confidence=concept.confidence, source_ids=source_ids,
        source_support=[{"source_id": link.source_id, "support_kind": link.support_kind} for link in source_links], created_at=concept.created_at,
        canonical_name=concept.canonical_name, definition=concept.definition, domain=concept.domain,
        mechanism=concept.mechanism, relationship_type=concept.relationship_type,
        relationship_evidence=concept.relationship_evidence, concept_verification=concept.concept_verification,
        relationship_verification=concept.relationship_verification, label_kind=concept.label_kind,
        curiosity_hook=concept.curiosity_hook, discovery_origin=concept.discovery_origin,
        research_path=concept.research_path, discovery_source_ids=json.loads(concept.discovery_source_ids_json or "[]"),
        initial_validation_outcome=concept.initial_validation_outcome,
        recovery_attempt=json.loads(concept.recovery_attempt_json or "{}"), final_validation_outcome=concept.final_validation_outcome,
        review_audit_id=concept.review_audit_id, user_note=concept.user_note,
        concept_classification=concept.concept_classification, evidence_form=concept.evidence_form,
        feedback_key=concept.feedback_key, discovery_mode=concept.discovery_mode,
        discovery_quality=json.loads(concept.discovery_quality_json or "{}"),
        appeal_dimensions=json.loads(concept.appeal_dimensions_json or "[]"), pattern_scope=concept.pattern_scope,
        spoiler_scope=concept.spoiler_scope,
        interested=concept.interested, interest_note=concept.interest_note, interested_at=concept.interested_at,
        loved=concept.loved, love_note=concept.love_note, loved_at=concept.loved_at,
        surprised=concept.surprised, surprise_note=concept.surprise_note, surprised_at=concept.surprised_at,
    )

def rejected_finding_read(lead: ConceptLeadAudit, db: Session) -> RejectedFindingRead:
    item = db.get(CuriosityItem, lead.item_id)
    snapshots = json.loads(lead.source_snapshot_json or "[]")
    known_urls = {source.get("url") for source in snapshots if isinstance(source, dict)}
    current_sources = db.scalars(select(ResearchSource).where(ResearchSource.item_id == lead.item_id).order_by(ResearchSource.id)).all()
    sources = list(snapshots)
    for source in current_sources:
        if source.url not in known_urls:
            sources.append({"source_id": f"saved-{source.id}", "id": source.id, "url": source.url, "title": source.title, "excerpt": source.excerpt, "source_type": source.source_type})
    return RejectedFindingRead(
        id=lead.id, item_id=lead.item_id, item_title=item.title if item else "Deleted curiosity",
        item_kind=Kind(item.kind) if item else Kind.thought, name=lead.name, stage=lead.stage,
        proposed_definition=lead.proposed_definition, motivating_detail=lead.motivating_detail, proposed_relationship=lead.proposed_relationship,
        relationship_type=lead.relationship_type, rejection_reason=lead.rejection_reason or "Rejected by quality review",
        discovery_origin=lead.discovery_origin, research_path=lead.research_path, research_questions=json.loads(lead.research_questions_json or "[]"),
        initial_validation_outcome=lead.initial_validation_outcome,
        recovery_attempt=json.loads(lead.recovery_attempt_json or "{}"), final_validation_outcome=lead.final_validation_outcome,
        automated_evidence_status=lead.automated_evidence_status, human_review_status=lead.human_review_status,
        human_review_note=lead.human_review_note, review_history=json.loads(lead.review_history_json or "[]"),
        kept_concept_id=lead.kept_concept_id, evidence_review_detail=lead.evidence_review_detail,
        last_evaluated_at=lead.last_evaluated_at, candidate=json.loads(lead.candidate_snapshot_json or "{}"), validation=json.loads(lead.validation_snapshot_json or "{}"),
        sources=sources, superseded_at=lead.superseded_at, created_at=lead.created_at,
        concept_classification=lead.concept_classification, evidence_form=lead.evidence_form,
        feedback_key=lead.feedback_key, discovery_mode=lead.discovery_mode,
        discovery_quality=json.loads(lead.discovery_quality_json or "{}"),
        interested=lead.interested, interest_note=lead.interest_note, interested_at=lead.interested_at,
        loved=lead.loved, love_note=lead.love_note, loved_at=lead.loved_at,
        surprised=lead.surprised, surprise_note=lead.surprise_note, surprised_at=lead.surprised_at,
    )

def enrichment_read(item_id: int, db: Session) -> EnrichmentRead:
    enrichment = db.get(ItemEnrichment, item_id)
    sources = db.scalars(select(ResearchSource).where(ResearchSource.item_id == item_id, ResearchSource.superseded_at.is_(None)).order_by(ResearchSource.id)).all()
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.superseded_at.is_(None)).order_by(ItemConcept.created_at, ItemConcept.id)).all()
    rejected = db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id, ConceptLeadAudit.accepted == False).order_by(ConceptLeadAudit.id)).all()  # noqa: E712
    return EnrichmentRead(
        status=enrichment.status if enrichment else "not_researched",
        resolved_title=enrichment.resolved_title if enrichment else None, creator=enrichment.creator if enrichment else None,
        release_info=enrichment.release_info if enrichment else None, identity_summary=enrichment.identity_summary if enrichment else None,
        confirmation_question=enrichment.confirmation_question if enrichment else None,
        identity_status=enrichment.identity_status if enrichment else "unresolved",
        identity_key=enrichment.identity_key if enrichment else None,
        resolved_media_type=enrichment.resolved_media_type if enrichment else None,
        resolved_work_scope=enrichment.resolved_work_scope if enrichment else None,
        parent_work_title=enrichment.parent_work_title if enrichment else None,
        canonical_identity_id=enrichment.canonical_identity_id if enrichment else None,
        identity_confidence=enrichment.identity_confidence if enrichment else None,
        identity_candidates=json.loads(enrichment.identity_candidates_json or "[]") if enrichment else [],
        identity_sources=json.loads(enrichment.identity_source_snapshot_json or "[]") if enrichment else [],
        confirmed_at=enrichment.confirmed_at if enrichment else None,
        researched_at=enrichment.researched_at if enrichment else None,
        sources=[ResearchSourceRead(
            id=source.id, url=source.url, title=source.title, publisher=source.publisher,
            source_type=source.source_type, evidence_roles=json.loads(source.evidence_roles_json or "[]"),
            excerpt=source.excerpt, retrieval_status=source.retrieval_status, retrieved_at=source.retrieved_at,
        ) for source in sources], concepts=[concept_read(concept, db) for concept in concepts],
        rejected_leads=[rejected_finding_read(lead, db) for lead in rejected],
    )


def normalize_concept_term(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


WORK_KINDS = {Kind.book.value, Kind.movie.value, Kind.tv_series.value, Kind.podcast.value, Kind.article.value}


def spoiler_ceiling(item: CuriosityItem) -> str:
    if item.spoiler_policy == SpoilerPolicy.full.value or item.progress_status in {ProgressStatus.finished.value, ProgressStatus.rereading.value, ProgressStatus.rewatching.value}:
        return "full"
    if item.spoiler_policy == SpoilerPolicy.through_progress.value and item.progress_current is not None:
        return "within_progress"
    return "premise_only"


def item_ai_payload(item: CuriosityItem) -> dict[str, object]:
    return {
        "kind": item.kind, "title": item.title, "note": item.note, "url": item.url,
        "creator": item.creator, "release_year": item.release_year,
        "work_scope": item.work_scope, "parent_work_title": item.parent_work_title,
        "canonical_identity_id": item.canonical_identity_id,
        "progress": {
            "status": item.progress_status, "current": item.progress_current,
            "total": item.progress_total, "unit": item.progress_unit,
        },
        "spoiler_policy": item.spoiler_policy, "spoiler_ceiling": spoiler_ceiling(item),
    }


def validate_progress_values(current: int | None, total: int | None, unit: str) -> None:
    if current is not None and total is not None and current > total:
        raise HTTPException(422, "Progress cannot be greater than the total.")
    if unit == ProgressUnit.percent.value and ((current is not None and current > 100) or (total is not None and total != 100)):
        raise HTTPException(422, "Percent progress must be between 0 and 100, with a total of 100 when supplied.")


def anchor_within_spoiler_scope(anchor, item: CuriosityItem) -> bool:
    order = {"premise_only": 0, "within_progress": 1, "full": 2}
    return order[anchor.spoiler_scope] <= order[spoiler_ceiling(item)]


def filter_work_anchors(
    problems: list[ResearchProblem], work_source_ids: set[str], item: CuriosityItem,
) -> tuple[list[ResearchProblem], list[dict[str, str]]]:
    """Keep independently valid anchors and return safe, inspectable discard reasons."""
    valid: list[ResearchProblem] = []
    discarded: list[dict[str, str]] = []
    seen_problem_ids: set[str] = set()
    for problem in problems:
        if problem.problem_id in seen_problem_ids:
            discarded.append({"problem_id": problem.problem_id, "category": "duplicate_id", "reason": "Duplicate work-anchor identifier."})
            continue
        seen_problem_ids.add(problem.problem_id)
        if reason := research_problem_rejection(problem, work_source_ids, item.title):
            discarded.append({"problem_id": problem.problem_id, "category": "quality", "reason": reason})
            continue
        if not anchor_within_spoiler_scope(problem, item):
            discarded.append({"problem_id": problem.problem_id, "category": "spoiler_boundary", "reason": "The proposed observation exceeds the saved spoiler boundary."})
            continue
        valid.append(problem)
    return valid, discarded


def identity_key(candidate: EntityCandidate) -> str:
    parts = [candidate.media_type, normalize_concept_term(candidate.title), normalize_concept_term(candidate.creator), str(candidate.release_year or "")]
    if candidate.work_scope != WorkScope.unknown.value:
        parts.extend((candidate.work_scope, normalize_concept_term(candidate.parent_work_title), candidate.canonical_identity_id.strip().casefold()))
    return "|".join(parts)


def identity_candidate_is_valid(candidate: EntityCandidate, item: CuriosityItem, source_ids: set[str]) -> bool:
    return (
        candidate.media_type == item.kind
        and normalize_concept_term(candidate.title) == normalize_concept_term(item.title)
        and candidate.confidence >= 0.90
        and bool(candidate.source_ids)
        and set(candidate.source_ids) <= source_ids
        and (not item.creator or normalize_concept_term(item.creator) in normalize_concept_term(candidate.creator))
        and (not item.release_year or item.release_year == candidate.release_year)
        and (item.work_scope == WorkScope.unknown.value or candidate.work_scope == item.work_scope)
        and (not item.parent_work_title or normalize_concept_term(item.parent_work_title) == normalize_concept_term(candidate.parent_work_title))
        and (not item.canonical_identity_id or item.canonical_identity_id.casefold() == candidate.canonical_identity_id.casefold())
    )


def save_identity_resolution(item: CuriosityItem, resolution, sources: list[dict[str, str]], db: Session) -> ItemEnrichment:
    enrichment = db.get(ItemEnrichment, item.id) or ItemEnrichment(item_id=item.id, status="not_researched")
    candidates = [candidate.model_dump() for candidate in resolution.candidates]
    enrichment.identity_candidates_json = json.dumps(candidates)
    enrichment.identity_source_snapshot_json = json.dumps(sources)
    enrichment.confirmation_question = resolution.reason
    valid = [candidate for candidate in resolution.candidates if identity_candidate_is_valid(candidate, item, {source["source_id"] for source in sources})]
    if resolution.status == "confirmed" and len(valid) == 1:
        candidate = valid[0]
        enrichment.identity_status = "confirmed"
        enrichment.identity_key = identity_key(candidate)
        enrichment.resolved_title = candidate.title
        enrichment.creator = candidate.creator or None
        enrichment.release_info = str(candidate.release_year) if candidate.release_year else None
        enrichment.resolved_media_type = candidate.media_type
        enrichment.resolved_work_scope = candidate.work_scope
        enrichment.parent_work_title = candidate.parent_work_title or None
        enrichment.canonical_identity_id = candidate.canonical_identity_id or None
        enrichment.identity_summary = candidate.description
        enrichment.identity_confidence = candidate.confidence
        enrichment.confirmed_at = datetime.now(timezone.utc)
        enrichment.confirmation_question = None
    else:
        enrichment.identity_status = "ambiguous" if resolution.candidates else "unresolved"
        enrichment.status = "needs_confirmation"
    db.add(enrichment)
    db.commit()
    db.refresh(enrichment)
    return enrichment


def retire_stale_identity_research(item_id: int, db: Session) -> None:
    now = datetime.now(timezone.utc)
    for concept in db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.origin.in_(("ai", "user_kept")), ItemConcept.superseded_at.is_(None))).all():
        concept.superseded_at = now
    for source in db.scalars(select(ResearchSource).where(ResearchSource.item_id == item_id, ResearchSource.superseded_at.is_(None))).all():
        source.superseded_at = now
    for audit in db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id, ConceptLeadAudit.superseded_at.is_(None))).all():
        audit.superseded_at = now


def concept_feedback_key(name: str, classification: str | None = None) -> str:
    material = f"concept|{classification or 'legacy'}|{normalize_concept_term(name)}"
    return hashlib.sha256(material.encode()).hexdigest()


def anchor_feedback_key(anchor) -> str:
    # Exact normalized anchor substance is deliberately conservative: a similar
    # title alone can never transfer personal feedback to a different finding.
    material = "|".join((
        "anchor", anchor.anchor_kind, normalize_concept_term(anchor.concrete_description),
        normalize_concept_term(anchor.work_situation), normalize_concept_term(anchor.interesting_reason),
    ))
    return hashlib.sha256(material.encode()).hexdigest()


def feedback_state(item_id: int, feedback_key: str, name: str, db: Session) -> dict[str, object]:
    """Carry feedback only across an exact fingerprint, with exact-name fallback for legacy rows."""
    normalized = normalize_concept_term(name)
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id)).all()
    audits = db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id)).all()
    entries = [*concepts, *audits]
    matches = [entry for entry in entries if entry.feedback_key == feedback_key]
    if not matches:
        # Legacy fallback is limited to earlier generated findings. A manually
        # authored concept with the same label is not evidence that the user
        # meant to endorse a newly generated observation.
        matches = [
            entry for entry in entries
            if entry.feedback_key is None
            and normalize_concept_term(entry.name) == normalized
            and (isinstance(entry, ConceptLeadAudit) or entry.origin != ConceptOrigin.user.value)
        ]

    def latest_active(attribute: str, timestamp: str, note: str) -> tuple[bool, str | None, datetime | None]:
        active = [entry for entry in matches if getattr(entry, attribute)]
        if not active:
            return False, None, None
        latest = max(active, key=lambda entry: getattr(entry, timestamp) or entry.created_at)
        return True, getattr(latest, note), getattr(latest, timestamp)

    interested, interest_note, interested_at = latest_active("interested", "interested_at", "interest_note")
    loved, love_note, loved_at = latest_active("loved", "loved_at", "love_note")
    surprised, surprise_note, surprised_at = latest_active("surprised", "surprised_at", "surprise_note")
    return {
        "feedback_key": feedback_key,
        "interested": interested, "interest_note": interest_note, "interested_at": interested_at,
        "loved": loved, "love_note": love_note, "loved_at": loved_at,
        "surprised": surprised, "surprise_note": surprise_note, "surprised_at": surprised_at,
    }


def feedback_matches(item_id: int, feedback_key: str | None, name: str, db: Session) -> list[ItemConcept | ConceptLeadAudit]:
    normalized = normalize_concept_term(name)
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id)).all()
    audits = db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id)).all()
    entries = [*concepts, *audits]
    if feedback_key:
        keyed = [entry for entry in entries if entry.feedback_key == feedback_key]
        if keyed:
            return keyed
    return [entry for entry in entries if entry.feedback_key is None and normalize_concept_term(entry.name) == normalized]


def synchronize_feedback(item_id: int, feedback_key: str | None, name: str, signal: str, active: bool, note: str | None, at: datetime | None, db: Session) -> None:
    fields = {
        "interest": ("interested", "interest_note", "interested_at"),
        "love": ("loved", "love_note", "loved_at"),
        "surprise": ("surprised", "surprise_note", "surprised_at"),
    }
    active_field, note_field, time_field = fields[signal]
    for entry in feedback_matches(item_id, feedback_key, name, db):
        setattr(entry, active_field, active)
        setattr(entry, note_field, note if active else None)
        setattr(entry, time_field, at if active else None)


def curiosity_profile(db: Session) -> dict[str, object]:
    """Derive editable-by-action working hypotheses from explicit positive signals only."""
    concepts = db.scalars(select(ItemConcept).order_by(ItemConcept.id.desc())).all()
    authored_lenses = []
    seen_authored: set[str] = set()
    for concept in concepts:
        if concept.origin == ConceptOrigin.user.value and concept.superseded_at is None:
            normalized = normalize_concept_term(concept.name)
            if normalized and normalized not in seen_authored:
                seen_authored.add(normalized)
                authored_lenses.append(concept.name)

    # One stable finding contributes once even if prior reruns remain in audit history.
    seen_feedback: set[str] = set()
    grouped: dict[str, dict[str, object]] = {}
    for concept in concepts:
        if not (concept.interested or concept.loved or concept.surprised):
            continue
        fingerprint = concept.feedback_key or f"concept:{concept.id}"
        if fingerprint in seen_feedback:
            continue
        seen_feedback.add(fingerprint)
        dimensions = json.loads(concept.appeal_dimensions_json or "[]")
        labels = dimensions or ([concept.domain] if concept.domain else [concept.name])
        weight = int(concept.interested) + 2 * int(concept.loved) + int(concept.surprised)
        notes = [note for note in (concept.interest_note, concept.love_note, concept.surprise_note) if note]
        for label in labels:
            key = normalize_concept_term(str(label))
            if not key:
                continue
            entry = grouped.setdefault(key, {"label": str(label).replace("_", " "), "score": 0, "signals": {"interested": 0, "love": 0, "surprise": 0}, "notes": []})
            entry["score"] = int(entry["score"]) + weight
            signals = entry["signals"]
            if isinstance(signals, dict):
                signals["interested"] = int(signals["interested"]) + int(concept.interested)
                signals["love"] = int(signals["love"]) + int(concept.loved)
                signals["surprise"] = int(signals["surprise"]) + int(concept.surprised)
            saved_notes = entry["notes"]
            if isinstance(saved_notes, list):
                for note in notes:
                    if note not in saved_notes and len(saved_notes) < 3:
                        saved_notes.append(note)
    hypotheses = sorted(grouped.values(), key=lambda entry: (-int(entry["score"]), str(entry["label"])))[:8]
    return {"hypotheses": hypotheses, "authored_lenses": authored_lenses[:8], "exploration_budget": 1}


def profile_terms(profile: dict[str, object]) -> set[str]:
    stop = {"and", "the", "with", "from", "into", "about", "this", "that", "concept", "general"}
    material = " ".join([
        *(str(entry.get("label", "")) for entry in profile.get("hypotheses", []) if isinstance(entry, dict)),
        *(str(value) for value in profile.get("authored_lenses", [])),
    ])
    return set(normalize_concept_term(material).split()) - stop


def discovery_limits() -> tuple[int, int]:
    selected = max(3, min(int(os.getenv("DISCOVERY_MAX_ANCHORS", "5")), 5))
    researched = max(0, min(int(os.getenv("DISCOVERY_MAX_RESEARCH_ANCHORS", "3")), selected))
    return selected, researched


def select_diverse_anchors(problems, personal_terms: set[str] | None = None) -> tuple[list[object], list[object]]:
    """Select strong anchors without another model call; keep the factors inspectable."""
    selected_limit, research_limit = discovery_limits()
    quality_order = {"high": 2, "medium": 1, "low": 0}

    personal_terms = personal_terms or set()

    def relevance(anchor) -> int:
        material = normalize_concept_term(" ".join((anchor.concise_label, anchor.mechanism, anchor.interesting_reason, *anchor.appeal_dimensions)))
        return min(2, len(set(material.split()) & personal_terms))

    def factors(anchor) -> tuple[int, int, int, int, int, int]:
        quality = anchor.discovery_quality
        return (
            quality_order[quality.specificity], quality_order[quality.intellectual_substance],
            relevance(anchor), quality_order[quality.surprise_potential], len(anchor.evidence_source_ids),
            len(anchor.interesting_reason.strip()),
        )

    ranked = sorted(problems, key=factors, reverse=True)
    selected: list[object] = []
    used_kinds: set[str] = set()
    used_modes: set[str] = set()
    while ranked and len(selected) < selected_limit:
        candidate = max(ranked, key=lambda anchor: (anchor.anchor_kind not in used_kinds, anchor.discovery_mode not in used_modes, factors(anchor)))
        ranked.remove(candidate)
        selected.append(candidate)
        used_kinds.add(candidate.anchor_kind)
        used_modes.add(candidate.discovery_mode)
    if personal_terms and selected and all(relevance(anchor) > 0 for anchor in selected):
        unfamiliar = next((anchor for anchor in ranked if relevance(anchor) == 0 and anchor.discovery_quality.specificity != "low" and anchor.discovery_quality.intellectual_substance != "low"), None)
        if unfamiliar is not None:
            selected[-1] = unfamiliar
    researchable = [anchor for anchor in selected if anchor.external_research_warranted][:research_limit]
    return selected, researchable


def evidence_recovery_limit() -> int:
    max_candidates = max(0, min(int(os.getenv("EVIDENCE_RECOVERY_MAX_CANDIDATES", "2")), 3))
    max_estimated_cost = max(0.0, float(os.getenv("EVIDENCE_RECOVERY_MAX_ESTIMATED_COST_USD", "0.03")))
    estimated_per_candidate = max(0.001, float(os.getenv("EVIDENCE_RECOVERY_ESTIMATED_COST_PER_CANDIDATE_USD", "0.015")))
    return min(max_candidates, int(max_estimated_cost / estimated_per_candidate))


def resolve_canonical_concept(db: Session, name: str, aliases: list[str], definition: str, domain: str, mechanism: str, verification_status: str, label_kind: str) -> CanonicalConcept:
    normalized_name = normalize_concept_term(name)
    canonical = db.scalar(select(CanonicalConcept).where(CanonicalConcept.normalized_name == normalized_name))
    if canonical is None and label_kind == "established_term":
        requested_terms = {normalized_name, *(normalize_concept_term(alias) for alias in aliases if alias.strip())}
        alias_matches = []
        for existing in db.scalars(select(CanonicalConcept)).all():
            existing_terms = {existing.normalized_name, *(normalize_concept_term(alias) for alias in json.loads(existing.aliases_json or "[]"))}
            if requested_terms & existing_terms:
                alias_matches.append(existing)
        if len(alias_matches) == 1:
            canonical = alias_matches[0]
    if canonical is None:
        canonical = CanonicalConcept(normalized_name=normalized_name, name=name.strip(), definition=definition, domain=domain, mechanism=mechanism, verification_status=verification_status, aliases_json=json.dumps(sorted(set(alias.strip() for alias in aliases if alias.strip()))), label_kind=label_kind, provenance="ai")
        db.add(canonical)
        db.flush()
    else:
        existing_aliases = set(json.loads(canonical.aliases_json or "[]"))
        verified_aliases = {alias.strip() for alias in aliases if alias.strip()} if label_kind == "established_term" else set()
        canonical.aliases_json = json.dumps(sorted(existing_aliases | verified_aliases))
    return canonical

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/discovery-profile", response_model=CuriosityProfileRead)
def get_discovery_profile(db: Session = Depends(get_db)):
    return curiosity_profile(db)

@app.get("/api/items", response_model=list[ItemRead])
def list_items(q: str = Query(default="", max_length=240), kind: Kind | None = None, db: Session = Depends(get_db)):
    stmt = select(CuriosityItem)
    if kind:
        stmt = stmt.where(CuriosityItem.kind == kind.value)
    if q.strip():
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(or_(CuriosityItem.title.ilike(pattern), CuriosityItem.note.ilike(pattern)))
    return db.scalars(stmt.order_by(CuriosityItem.created_at.desc(), CuriosityItem.id.desc())).all()

@app.post("/api/items", response_model=ItemRead, status_code=201)
def create_item(data: ItemCreate, db: Session = Depends(get_db)):
    if not data.title.strip():
        raise HTTPException(422, "Title cannot be blank")
    validate_progress_values(data.progress_current, data.progress_total, data.progress_unit.value)
    item = CuriosityItem(**{**data.model_dump(), "title": data.title.strip()})
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@app.get("/api/items/{item_id}/concepts", response_model=list[ConceptRead])
def list_concepts(item_id: int, db: Session = Depends(get_db)):
    if db.get(CuriosityItem, item_id) is None:
        raise HTTPException(404, "Item not found")
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.superseded_at.is_(None)).order_by(ItemConcept.created_at, ItemConcept.id)).all()
    return [concept_read(concept, db) for concept in concepts]

@app.post("/api/items/{item_id}/concepts", response_model=ConceptRead, status_code=201)
def add_concept(item_id: int, data: ConceptCreate, db: Session = Depends(get_db)):
    if db.get(CuriosityItem, item_id) is None:
        raise HTTPException(404, "Item not found")
    if not data.name.strip():
        raise HTTPException(422, "Concept name cannot be blank")
    concept = ItemConcept(item_id=item_id, name=data.name.strip(), category=data.category.value, origin="user", explanation=data.explanation)
    db.add(concept)
    db.commit()
    db.refresh(concept)
    return concept

@app.delete("/api/concepts/{concept_id}", status_code=204)
def delete_concept(concept_id: int, db: Session = Depends(get_db)):
    concept = db.get(ItemConcept, concept_id)
    if concept is None:
        raise HTTPException(404, "Concept not found")
    if concept.review_audit_id is not None:
        lead = db.get(ConceptLeadAudit, concept.review_audit_id)
        if lead is not None:
            history = json.loads(lead.review_history_json or "[]")
            history.append({"decision": "pending", "note": "User removed the kept concept.", "at": datetime.now(timezone.utc).isoformat()})
            lead.human_review_status = "pending"
            lead.kept_concept_id = None
            lead.review_history_json = json.dumps(history)
    db.delete(concept)
    db.commit()

@app.get("/api/rejected-findings", response_model=list[RejectedFindingRead])
def list_rejected_findings(item_id: int | None = None, review_status: Literal["pending", "kept", "dismissed"] | None = None, db: Session = Depends(get_db)):
    stmt = select(ConceptLeadAudit).where(ConceptLeadAudit.accepted == False).order_by(ConceptLeadAudit.created_at.desc(), ConceptLeadAudit.id.desc())  # noqa: E712
    if item_id is not None:
        stmt = stmt.where(ConceptLeadAudit.item_id == item_id)
    if review_status is not None:
        stmt = stmt.where(ConceptLeadAudit.human_review_status == review_status)
    return [rejected_finding_read(lead, db) for lead in db.scalars(stmt).all()]

def update_kept_concept_status(lead: ConceptLeadAudit, db: Session) -> None:
    if lead.kept_concept_id is None:
        return
    concept = db.get(ItemConcept, lead.kept_concept_id)
    if concept is not None:
        concept.evidence_status = f"user_kept_{lead.automated_evidence_status}"
        concept.concept_verification = "partially_verified" if lead.automated_evidence_status == "partially_supported" else "unverified"

@app.patch("/api/rejected-findings/{finding_id}/review", response_model=RejectedFindingRead)
def review_rejected_finding(finding_id: int, data: ReviewDecision, db: Session = Depends(get_db)):
    lead = db.get(ConceptLeadAudit, finding_id)
    if lead is None or lead.accepted:
        raise HTTPException(404, "Rejected finding not found")
    if lead.stage != "concept_candidate" and data.decision == "keep":
        raise HTTPException(422, "Only rejected concept candidates can be kept as discoveries.")
    now = datetime.now(timezone.utc)
    history = json.loads(lead.review_history_json or "[]")
    history.append({"decision": data.decision, "note": data.note, "at": now.isoformat()})
    existing = db.get(ItemConcept, lead.kept_concept_id) if lead.kept_concept_id else db.scalar(select(ItemConcept).where(ItemConcept.review_audit_id == lead.id))
    if data.decision == "keep":
        candidate = json.loads(lead.candidate_snapshot_json or "{}")
        if existing is None:
            existing = ItemConcept(
                item_id=lead.item_id, name=lead.name.strip(), category="mechanism" if candidate.get("level") == "mechanism" else "general",
                origin="user_kept", explanation=lead.proposed_definition or None, definition=lead.proposed_definition or None,
                domain=candidate.get("domain"), mechanism=candidate.get("mechanism"), level=candidate.get("level") or "concept",
                relationship=lead.proposed_relationship or None, relationship_type=lead.relationship_type,
                relationship_evidence=lead.proposed_relationship or None, evidence_status=f"user_kept_{lead.automated_evidence_status}",
                concept_verification="partially_verified" if lead.automated_evidence_status == "partially_supported" else "unverified",
                relationship_verification="unverified", label_kind=candidate.get("label_kind"), curiosity_hook=candidate.get("curiosity_hook"),
                discovery_origin=lead.discovery_origin, research_path=lead.research_path,
                discovery_source_ids_json=lead.discovery_source_ids_json or "[]", initial_validation_outcome=lead.initial_validation_outcome,
                recovery_attempt_json=lead.recovery_attempt_json or "{}", final_validation_outcome=lead.final_validation_outcome or lead.rejection_reason,
                review_audit_id=lead.id, user_note=data.note,
                concept_classification=lead.concept_classification or "speculative_connection",
                evidence_form=lead.evidence_form, feedback_key=lead.feedback_key,
                discovery_mode=lead.discovery_mode, discovery_quality_json=lead.discovery_quality_json or "{}",
                interested=lead.interested, interest_note=lead.interest_note, interested_at=lead.interested_at,
                loved=lead.loved, love_note=lead.love_note, loved_at=lead.loved_at,
                surprised=lead.surprised, surprise_note=lead.surprise_note, surprised_at=lead.surprised_at,
            )
            db.add(existing)
            db.flush()
        else:
            existing.user_note = data.note
        lead.kept_concept_id = existing.id
    elif existing is not None:
        db.execute(delete(ConceptSource).where(ConceptSource.concept_id == existing.id))
        db.delete(existing)
        lead.kept_concept_id = None
    lead.human_review_status = {"keep": "kept", "dismiss": "dismissed", "pending": "pending"}[data.decision]
    lead.human_review_note = data.note
    lead.review_history_json = json.dumps(history)
    db.commit()
    db.refresh(lead)
    return rejected_finding_read(lead, db)


@app.patch("/api/concepts/{concept_id}/interest", response_model=ConceptRead)
def update_concept_interest(concept_id: int, data: InterestUpdate, db: Session = Depends(get_db)):
    concept = db.get(ItemConcept, concept_id)
    if concept is None:
        raise HTTPException(404, "Concept not found")
    at = datetime.now(timezone.utc) if data.interested else None
    synchronize_feedback(concept.item_id, concept.feedback_key, concept.name, "interest", data.interested, data.note, at, db)
    db.commit()
    db.refresh(concept)
    return concept_read(concept, db)


@app.patch("/api/rejected-findings/{finding_id}/interest", response_model=RejectedFindingRead)
def update_finding_interest(finding_id: int, data: InterestUpdate, db: Session = Depends(get_db)):
    lead = db.get(ConceptLeadAudit, finding_id)
    if lead is None or lead.accepted:
        raise HTTPException(404, "Rejected finding not found")
    at = datetime.now(timezone.utc) if data.interested else None
    synchronize_feedback(lead.item_id, lead.feedback_key, lead.name, "interest", data.interested, data.note, at, db)
    db.commit()
    db.refresh(lead)
    return rejected_finding_read(lead, db)


@app.patch("/api/concepts/{concept_id}/feedback", response_model=ConceptRead)
def update_concept_feedback(concept_id: int, data: DiscoveryFeedbackUpdate, db: Session = Depends(get_db)):
    concept = db.get(ItemConcept, concept_id)
    if concept is None:
        raise HTTPException(404, "Concept not found")
    at = datetime.now(timezone.utc) if data.active else None
    synchronize_feedback(concept.item_id, concept.feedback_key, concept.name, data.signal, data.active, data.note, at, db)
    db.commit()
    db.refresh(concept)
    return concept_read(concept, db)


@app.patch("/api/rejected-findings/{finding_id}/feedback", response_model=RejectedFindingRead)
def update_finding_feedback(finding_id: int, data: DiscoveryFeedbackUpdate, db: Session = Depends(get_db)):
    lead = db.get(ConceptLeadAudit, finding_id)
    if lead is None or lead.accepted:
        raise HTTPException(404, "Rejected finding not found")
    at = datetime.now(timezone.utc) if data.active else None
    synchronize_feedback(lead.item_id, lead.feedback_key, lead.name, data.signal, data.active, data.note, at, db)
    db.commit()
    db.refresh(lead)
    return rejected_finding_read(lead, db)

@app.post("/api/rejected-findings/{finding_id}/reevaluate", response_model=RejectedFindingRead)
def reevaluate_rejected_finding(finding_id: int, db: Session = Depends(get_db)):
    lead = db.get(ConceptLeadAudit, finding_id)
    if lead is None or lead.accepted:
        raise HTTPException(404, "Rejected finding not found")
    snapshot = json.loads(lead.candidate_snapshot_json or "{}")
    now = datetime.now(timezone.utc)
    if not snapshot:
        lead.automated_evidence_status = "unverified"
        lead.evidence_review_detail = "This legacy finding has no structured candidate snapshot; its original rejection remains unchanged. Refreshing research can produce a newly reviewable finding."
    else:
        sources = rejected_finding_read(lead, db).sources
        texts = {str(source.get("source_id")): f"{source.get('title', '')} {source.get('excerpt', '')}" for source in sources}
        urls = {str(source.get("source_id")): str(source.get("url", "")) for source in sources}
        try:
            candidate = EvidenceConceptCandidate.model_validate(snapshot)
            candidate, events = rebind_exact_evidence(candidate, texts)
            reason = candidate_evidence_rejection(candidate, texts, urls)
            if reason is None:
                lead.automated_evidence_status = "partially_supported"
                lead.evidence_review_detail = "Independent concept evidence now passes deterministic source and quotation checks. The original relationship rejection and automated outcome remain unchanged."
            else:
                lead.automated_evidence_status = "unverified"
                lead.evidence_review_detail = reason
            if events:
                lead.evidence_review_detail += f" Rebound {len(events)} exact excerpt reference(s) during review."
        except ValueError:
            lead.automated_evidence_status = "unverified"
            lead.evidence_review_detail = "The stored candidate snapshot is incomplete and could not be re-evaluated."
    lead.last_evaluated_at = now
    update_kept_concept_status(lead, db)
    db.commit()
    db.refresh(lead)
    return rejected_finding_read(lead, db)

@app.post("/api/items/{item_id}/concepts/extract", response_model=list[ConceptRead])
def extract_concepts(item_id: int, db: Session = Depends(get_db), ai: OpenRouterClient = Depends(get_ai_client)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    try:
        payload = item_ai_payload(item)
        payload["curiosity_profile"] = curiosity_profile(db)
        extraction = ai.extract_concepts(payload)
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    now = datetime.now(timezone.utc)
    for previous in db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.origin == "ai", ItemConcept.superseded_at.is_(None))).all():
        previous.superseded_at = now
    for extracted in extraction.concepts:
        key = concept_feedback_key(extracted.name, "quick_extraction")
        feedback = feedback_state(item_id, key, extracted.name, db)
        db.add(ItemConcept(item_id=item_id, name=extracted.name.strip(), category=extracted.category, origin="ai", explanation=extracted.explanation, **feedback))
    db.commit()
    return db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.superseded_at.is_(None)).order_by(ItemConcept.created_at, ItemConcept.id)).all()

@app.get("/api/items/{item_id}/enrichment", response_model=EnrichmentRead)
def get_enrichment(item_id: int, db: Session = Depends(get_db)):
    if db.get(CuriosityItem, item_id) is None:
        raise HTTPException(404, "Item not found")
    return enrichment_read(item_id, db)


@app.post("/api/items/{item_id}/identity/resolve", response_model=EnrichmentRead)
def resolve_item_identity(item_id: int, db: Session = Depends(get_db), ai: OpenRouterClient = Depends(get_ai_client), research: ResearchProvider = Depends(get_research_provider)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    existing = db.get(ItemEnrichment, item_id)
    if existing is not None and existing.identity_status == "confirmed":
        return enrichment_read(item_id, db)
    if item.kind not in WORK_KINDS:
        enrichment = db.get(ItemEnrichment, item_id) or ItemEnrichment(item_id=item_id, status="not_researched")
        enrichment.identity_status = "not_applicable"
        enrichment.resolved_title = item.title
        db.add(enrichment)
        db.commit()
        return enrichment_read(item_id, db)
    payload = item_ai_payload(item)
    try:
        packet = research.research(payload)
        source_payloads = [{"source_id": f"identity-{index}", "title": source.title, "excerpt": source.excerpt[:2500], "url": source.url} for index, source in enumerate(packet.sources, 1)]
        resolution = ai.resolve_identity(payload, packet.synthesis, [{key: source[key] for key in ("source_id", "title", "excerpt")} for source in source_payloads])
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    save_identity_resolution(item, resolution, source_payloads, db)
    return enrichment_read(item_id, db)


@app.post("/api/items/{item_id}/identity/confirm", response_model=EnrichmentRead)
def confirm_item_identity(item_id: int, data: IdentityConfirmation, db: Session = Depends(get_db)):
    item = db.get(CuriosityItem, item_id)
    enrichment = db.get(ItemEnrichment, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    if enrichment is None:
        raise HTTPException(409, "Resolve this work before confirming its identity.")
    candidates = [EntityCandidate.model_validate(candidate) for candidate in json.loads(enrichment.identity_candidates_json or "[]")]
    candidate = next((entry for entry in candidates if entry.candidate_id == data.candidate_id), None)
    if candidate is None:
        raise HTTPException(422, "That identity candidate is no longer available. Resolve the work again.")
    if candidate.media_type != item.kind:
        raise HTTPException(422, "The selected identity does not match this curiosity's media type.")
    if item.work_scope != WorkScope.unknown.value and candidate.work_scope != item.work_scope:
        raise HTTPException(422, "The selected identity does not match this curiosity's work scope.")
    changed = enrichment.identity_key != identity_key(candidate) and (enrichment.identity_key is not None or enrichment.status == "researched")
    if changed:
        retire_stale_identity_research(item_id, db)
    enrichment.identity_status = "confirmed"
    enrichment.identity_key = identity_key(candidate)
    enrichment.resolved_title = candidate.title
    enrichment.creator = candidate.creator or None
    enrichment.release_info = str(candidate.release_year) if candidate.release_year else None
    enrichment.resolved_media_type = candidate.media_type
    enrichment.resolved_work_scope = candidate.work_scope
    enrichment.parent_work_title = candidate.parent_work_title or None
    enrichment.canonical_identity_id = candidate.canonical_identity_id or None
    enrichment.identity_summary = candidate.description
    enrichment.identity_confidence = candidate.confidence
    enrichment.confirmed_at = datetime.now(timezone.utc)
    enrichment.confirmation_question = None
    enrichment.status = "not_researched"
    db.commit()
    return enrichment_read(item_id, db)


@app.delete("/api/items/{item_id}/identity", response_model=EnrichmentRead)
def reset_item_identity(item_id: int, db: Session = Depends(get_db)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    enrichment = db.get(ItemEnrichment, item_id) or ItemEnrichment(item_id=item_id, status="not_researched")
    retire_stale_identity_research(item_id, db)
    enrichment.status = "not_researched"
    enrichment.identity_status = "unresolved"
    enrichment.identity_key = None
    enrichment.resolved_title = None
    enrichment.creator = None
    enrichment.release_info = None
    enrichment.resolved_media_type = None
    enrichment.resolved_work_scope = None
    enrichment.parent_work_title = None
    enrichment.canonical_identity_id = None
    enrichment.identity_summary = None
    enrichment.identity_confidence = None
    enrichment.identity_candidates_json = "[]"
    enrichment.identity_source_snapshot_json = "[]"
    enrichment.confirmed_at = None
    enrichment.confirmation_question = None
    db.add(enrichment)
    db.commit()
    return enrichment_read(item_id, db)

@app.post("/api/items/{item_id}/enrich", response_model=EnrichmentRead)
def enrich_item(item_id: int, data: EnrichmentRequest | None = None, db: Session = Depends(get_db), ai: OpenRouterClient = Depends(get_ai_client), research: ResearchProvider = Depends(get_research_provider)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    payload = item_ai_payload(item)
    profile = curiosity_profile(db)
    payload["curiosity_profile"] = profile
    enrichment = db.get(ItemEnrichment, item_id)
    try:
        packet = None
        if item.kind in WORK_KINDS and (enrichment is None or enrichment.identity_status != "confirmed"):
            prior_identity_key = enrichment.identity_key if enrichment else None
            had_prior_research = bool(enrichment and enrichment.status == "researched")
            identity_packet = research.research(payload)
            identity_sources = [{"source_id": f"identity-{index}", "title": source.title, "excerpt": source.excerpt[:2500], "url": source.url} for index, source in enumerate(identity_packet.sources, 1)]
            resolution = ai.resolve_identity(payload, identity_packet.synthesis, [{key: source[key] for key in ("source_id", "title", "excerpt")} for source in identity_sources])
            enrichment = save_identity_resolution(item, resolution, identity_sources, db)
            if enrichment.identity_status != "confirmed":
                return enrichment_read(item_id, db)
            if had_prior_research and prior_identity_key != enrichment.identity_key:
                retire_stale_identity_research(item_id, db)
                db.commit()
            packet = identity_packet
        if enrichment is not None and enrichment.identity_status == "confirmed":
            payload["confirmed_identity"] = {
                "identity_key": enrichment.identity_key, "title": enrichment.resolved_title,
                "creator": enrichment.creator, "release_info": enrichment.release_info,
                "media_type": enrichment.resolved_media_type, "work_scope": enrichment.resolved_work_scope,
                "parent_work_title": enrichment.parent_work_title,
                "canonical_identity_id": enrichment.canonical_identity_id,
            }
        if packet is None:
            packet = research.research(payload)
        work_source_payloads = [{"source_id": f"work-{index}", "title": source.title, "excerpt": source.excerpt[:2500]} for index, source in enumerate(packet.sources, 1)]
        work_source_urls = {payload["source_id"]: source.url for payload, source in zip(work_source_payloads, packet.sources)}
        understanding = ai.understand_work(payload, packet.synthesis, work_source_payloads)
        understanding = understanding.model_copy(update={"spoiler_ceiling": spoiler_ceiling(item)})
        if item.kind in {Kind.concept.value, Kind.question.value, Kind.thought.value}:
            understanding = understanding.model_copy(update={"ambiguous_identity": False, "confirmation_question": "", "resolved_title": understanding.resolved_title or item.title, "identity_matches_sources": True})
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc

    if item.kind in WORK_KINDS and (understanding.ambiguous_identity or not understanding.identity_matches_sources):
        reason = understanding.identity_mismatch_reason or understanding.confirmation_question or "The retrieved sources do not match the confirmed work identity."
        raise HTTPException(502, f"{reason} Existing research was preserved; correct or resolve the identity before retrying.")

    selected_anchors = []
    research_anchors = []
    all_concept_sources = []
    concept_source_payloads: list[dict[str, str]] = []
    concept_source_urls: dict[str, str] = {}
    concept_source_text: dict[str, str] = {}
    candidates = EvidenceConceptBatch(candidates=[], insufficient_evidence_reasons=[])
    validation = ConceptValidationBatch(results=[])
    recovery_attempts: dict[str, dict[str, object]] = {}
    initial_evidence_outcomes: dict[str, str] = {}
    recovery_source_urls: set[str] = set()
    external_research_error: str | None = None
    try:
        proposed_anchors = ai.identify_research_problems(understanding, work_source_payloads, profile)
        work_source_ids = set(work_source_urls)
        valid_anchors, discarded_anchors = filter_work_anchors(proposed_anchors.problems, work_source_ids, item)
        if discarded_anchors:
            categories: dict[str, int] = {}
            for discarded in discarded_anchors:
                category = discarded["category"]
                categories[category] = categories.get(category, 0) + 1
            logger.warning(
                "Discarded invalid work anchors: item_id=%s kept=%s discarded=%s categories=%s",
                item.id, len(valid_anchors), len(discarded_anchors), categories,
            )
        if not valid_anchors:
            reasons = "; ".join(dict.fromkeys(discarded["reason"] for discarded in discarded_anchors))
            raise AIProviderError(
                f"No usable work anchors were generated. {reasons} "
                "Existing research was preserved."
            )

        selected_anchors, research_anchors = select_diverse_anchors(
            valid_anchors, profile_terms(profile)
        )
        # The internal research subset may intentionally be empty: grounded
        # observations do not need external research to be worth preserving.
        problems = ResearchProblemBatch.model_construct(problems=research_anchors)
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc

    try:
        if not research_anchors:
            raise StopIteration
        concept_packet = research.research_problems(payload, [problem.model_dump() for problem in research_anchors])
        all_concept_sources = list(concept_packet.sources)
        concept_source_payloads = [{"source_id": f"concept-{index}", "title": source.title, "excerpt": source.excerpt[:2200]} for index, source in enumerate(all_concept_sources, 1)]
        concept_source_urls = {source_payload["source_id"]: source.url for source_payload, source in zip(concept_source_payloads, all_concept_sources)}
        concept_source_text = {source_payload["source_id"]: f"{source.title} {source.excerpt}" for source_payload, source in zip(concept_source_payloads, all_concept_sources)}
        candidates = ai.extract_evidence_concepts(problems, concept_packet.synthesis, concept_source_payloads)
        recovered_candidates = []
        targeted_attempts = 0
        recovery_limit = evidence_recovery_limit()
        for candidate in candidates.candidates:
            initial_reason = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
            initial_evidence_outcomes[candidate.candidate_id] = initial_reason or "Initial evidence preflight passed."
            candidate, exact_events = rebind_exact_evidence(candidate, concept_source_text)
            remaining_reason = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
            attempt: dict[str, object] = {"initial_issue": initial_reason, "existing_corpus_rebindings": exact_events, "existing_corpus_review_attempted": False, "targeted_retrieval_attempted": False, "outcome": remaining_reason or "Existing corpus evidence passed."}
            if remaining_reason and targeted_attempts < recovery_limit:
                targeted_attempts += 1
                try:
                    attempt["existing_corpus_review_attempted"] = True
                    corpus_recovery = ai.recover_candidate_evidence(candidate, concept_packet.synthesis, concept_source_payloads)
                    if corpus_recovery.candidate_id != candidate.candidate_id:
                        raise AIProviderError("Existing-corpus evidence review returned the wrong candidate identifier.")
                    if corpus_recovery.recovered:
                        candidate = candidate.model_copy(update={"concept_evidence": corpus_recovery.concept_evidence})
                    remaining_reason = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
                    attempt.update({"existing_corpus_review_reason": corpus_recovery.reason, "outcome": remaining_reason or "Existing-corpus semantic evidence recovery passed."})
                    if remaining_reason:
                        attempt["targeted_retrieval_attempted"] = True
                        recovery_packet = research.recover_candidate(payload, candidate.model_dump())
                        recovery_payloads = []
                        for index, source in enumerate(recovery_packet.sources, 1):
                            source_id = f"recovery-{targeted_attempts}-{index}"
                            source_payload = {"source_id": source_id, "title": source.title, "excerpt": source.excerpt[:2200]}
                            recovery_payloads.append(source_payload)
                            concept_source_payloads.append(source_payload)
                            all_concept_sources.append(source)
                            concept_source_urls[source_id] = source.url
                            concept_source_text[source_id] = f"{source.title} {source.excerpt}"
                            recovery_source_urls.add(source.url)
                        recovery = ai.recover_candidate_evidence(candidate, recovery_packet.synthesis, recovery_payloads)
                        if recovery.candidate_id != candidate.candidate_id:
                            raise AIProviderError("Evidence recovery returned the wrong candidate identifier.")
                        if recovery.recovered:
                            candidate = candidate.model_copy(update={"concept_evidence": recovery.concept_evidence})
                        final_evidence_issue = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
                        attempt.update({"recovery_reason": recovery.reason, "recovery_source_ids": [source_payload["source_id"] for source_payload in recovery_payloads], "outcome": final_evidence_issue or "Targeted evidence recovery passed."})
                except (AIConfigurationError, AIProviderError) as exc:
                    attempt.update({"recovery_reason": str(exc), "outcome": f"Evidence recovery failed: {exc}"})
            recovery_attempts[candidate.candidate_id] = attempt
            recovered_candidates.append(candidate)
        candidates = candidates.model_copy(update={"candidates": recovered_candidates})
        validation = ai.validate_concepts(understanding, problems, candidates, work_source_payloads, concept_source_payloads)
    except StopIteration:
        pass
    except (AIConfigurationError, AIProviderError) as exc:
        external_research_error = str(exc)
        all_concept_sources = []
        concept_source_payloads = []
        concept_source_urls = {}
        concept_source_text = {}
        candidates = EvidenceConceptBatch(candidates=[], insufficient_evidence_reasons=[f"External concept research was unavailable: {exc}"])
        validation = ConceptValidationBatch(results=[])

    retire_stale_identity_research(item_id, db)
    unique_sources = {source.url: (source, "work") for source in packet.sources if source.url}
    for source in all_concept_sources:
        if source.url:
            source_type = "evidence_recovery" if source.url in recovery_source_urls else "concept_research"
            unique_sources[source.url] = (source, "work_and_concept" if source.url in unique_sources else source_type)
    saved_sources: dict[str, ResearchSource] = {}
    for source, source_type in unique_sources.values():
        saved = db.scalar(select(ResearchSource).where(ResearchSource.item_id == item_id, ResearchSource.url == source.url))
        if saved is None:
            roles = []
            if source_type in {"work", "work_and_concept"}:
                roles.extend(("work_identity", "work_detail"))
            if source_type in {"concept_research", "evidence_recovery", "work_and_concept"}:
                roles.append("external_concept")
            saved = ResearchSource(item_id=item_id, url=source.url, title=source.title[:500], publisher=urlparse(source.url).netloc or None, source_type=source_type, evidence_roles_json=json.dumps(roles), excerpt=source.excerpt[:2500])
            db.add(saved)
        else:
            saved.title = source.title[:500]
            saved.publisher = urlparse(source.url).netloc or None
            saved.source_type = source_type
            roles = set(json.loads(saved.evidence_roles_json or "[]"))
            roles.update(("work_identity", "work_detail") if source_type in {"work", "work_and_concept"} else ())
            roles.update(("external_concept",) if source_type in {"concept_research", "evidence_recovery", "work_and_concept"} else ())
            saved.evidence_roles_json = json.dumps(sorted(roles))
            saved.excerpt = source.excerpt[:2500]
            saved.retrieval_status = "retrieved"
            saved.retrieved_at = datetime.now(timezone.utc)
            saved.superseded_at = None
        db.flush()
        saved_sources[source.url] = saved
    enrichment = db.get(ItemEnrichment, item_id) or ItemEnrichment(item_id=item_id, status="researched")
    enrichment.status = "researched"
    if item.kind not in WORK_KINDS:
        enrichment.identity_status = "not_applicable"
        enrichment.resolved_title = understanding.resolved_title or item.title
        enrichment.creator = understanding.creator or None
        enrichment.release_info = understanding.release_info or None
        enrichment.resolved_media_type = item.kind
    enrichment.identity_summary = understanding.identity_summary or None
    enrichment.confirmation_question = understanding.confirmation_question or None
    enrichment.researched_at = datetime.now(timezone.utc)
    db.add(enrichment)
    work_source_ids = set(work_source_urls)
    work_payload_by_id = {source["source_id"]: source for source in work_source_payloads}
    seen_observations: set[str] = set()
    for anchor in selected_anchors:
        label = (anchor.concise_label or anchor.intellectual_question).strip()[:120]
        key = anchor_feedback_key(anchor)
        if key in seen_observations:
            continue
        seen_observations.add(key)
        feedback = feedback_state(item_id, key, label, db)
        quality_json = json.dumps(anchor.discovery_quality.model_dump())
        evidence_snapshots = [
            {**work_payload_by_id[source_id], "url": work_source_urls.get(source_id, ""), "source_type": "work"}
            for source_id in anchor.evidence_source_ids if source_id in work_payload_by_id
        ]
        audit = ConceptLeadAudit(
            item_id=item_id, name=label, proposed_definition=anchor.concrete_description,
            motivating_detail=anchor.work_situation, proposed_relationship=anchor.interesting_reason,
            relationship_type="explicit", research_questions_json=json.dumps(anchor.research_queries),
            accepted=True, stage="work_anchor", discovery_origin="work_grounded",
            discovery_source_ids_json=json.dumps(anchor.evidence_source_ids), research_path=anchor.distinctiveness,
            initial_validation_outcome="Work-source evidence and anchor specificity passed deterministic checks.",
            final_validation_outcome=external_research_error or "Grounded observation accepted independently of external concept research.",
            candidate_snapshot_json=json.dumps(anchor.model_dump()), validation_snapshot_json="{}",
            source_snapshot_json=json.dumps(evidence_snapshots), automated_evidence_status="partially_supported",
            evidence_review_detail="The observation is supported by work-specific evidence; no claim of established external terminology is made.",
            concept_classification="grounded_observation", evidence_form="paraphrase",
            discovery_mode=anchor.discovery_mode, discovery_quality_json=quality_json, **feedback,
        )
        db.add(audit)
        db.flush()
        observation = ItemConcept(
            item_id=item_id, name=label, canonical_name=label, category="mechanism", origin="ai",
            explanation=anchor.concrete_description, definition=anchor.concrete_description,
            domain=anchor.anchor_kind.replace("_", " "), mechanism=anchor.mechanism or anchor.interesting_reason,
            level="mechanism", relationship=anchor.work_situation, relationship_type="explicit",
            relationship_evidence=anchor.work_situation, evidence_status="work_supported",
            concept_verification="work_supported", relationship_verification="verified",
            label_kind="descriptive_label", curiosity_hook=anchor.curiosity_invitation or anchor.intellectual_question,
            discovery_origin="work_grounded", research_path=anchor.distinctiveness,
            discovery_source_ids_json=json.dumps(anchor.evidence_source_ids),
            initial_validation_outcome="Work-source evidence and anchor specificity passed deterministic checks.",
            final_validation_outcome="Grounded observation accepted independently of external concept research.",
            review_audit_id=audit.id, concept_classification="grounded_observation", evidence_form="paraphrase",
            discovery_mode=anchor.discovery_mode, discovery_quality_json=quality_json,
            appeal_dimensions_json=json.dumps(anchor.appeal_dimensions), pattern_scope=anchor.pattern_scope,
            spoiler_scope=anchor.spoiler_scope, **feedback,
        )
        db.add(observation)
        db.flush()
        for source_id in anchor.evidence_source_ids:
            url = work_source_urls.get(source_id)
            if url and url in saved_sources:
                db.add(ConceptSource(concept_id=observation.id, source_id=saved_sources[url].id, support_kind="observation"))
    for result in validation.results:
        if result.accepted and (quality_reason := concept_quality_rejection(result, work_source_ids, concept_source_text, concept_source_urls, item.title)):
            result.accepted = False
            result.rejection_reason = quality_reason
    results_by_candidate = {result.candidate_id: result for result in validation.results}
    for candidate in candidates.candidates:
        result = results_by_candidate.get(candidate.candidate_id)
        motivating_detail = "\n".join(reference.excerpt for reference in candidate.concept_evidence)
        final_outcome = "Accepted after final validation." if result and result.accepted else (result.rejection_reason if result else "Relationship validation returned no result.")
        referenced_ids = set(candidate.discovery_source_ids) | {reference.source_id for reference in candidate.concept_evidence} | set(result.relationship_source_ids if result else [])
        source_payload_by_id = {source["source_id"]: source for source in [*work_source_payloads, *concept_source_payloads]}
        source_url_by_id = {**work_source_urls, **concept_source_urls}
        source_snapshots = [{**source_payload_by_id[source_id], "url": source_url_by_id.get(source_id, ""), "source_type": "work" if source_id in work_source_ids else "independent"} for source_id in referenced_ids if source_id in source_payload_by_id]
        evidence_issue = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
        key = concept_feedback_key(candidate.canonical_name, candidate.concept_classification)
        feedback = feedback_state(item_id, key, candidate.canonical_name, db)
        db.add(ConceptLeadAudit(item_id=item_id, name=candidate.canonical_name, proposed_definition=candidate.definition, motivating_detail=motivating_detail, proposed_relationship=result.relationship_evidence if result else candidate.research_path, relationship_type=result.relationship_type if result else "unsupported", research_questions_json=json.dumps(candidate.problem_ids), accepted=bool(result and result.accepted), rejection_reason=None if result and result.accepted else final_outcome, stage="concept_candidate", discovery_origin=candidate.discovery_origin, discovery_source_ids_json=json.dumps(candidate.discovery_source_ids), research_path=candidate.research_path, initial_validation_outcome=initial_evidence_outcomes.get(candidate.candidate_id), recovery_attempt_json=json.dumps(recovery_attempts.get(candidate.candidate_id, {})), final_validation_outcome=final_outcome, candidate_snapshot_json=json.dumps(candidate.model_dump()), validation_snapshot_json=json.dumps(result.model_dump()) if result else "{}", source_snapshot_json=json.dumps(source_snapshots), automated_evidence_status="partially_supported" if evidence_issue is None else "unverified", evidence_review_detail=evidence_issue or "Independent concept evidence passed deterministic checks; the final automated outcome is preserved separately.", concept_classification=candidate.concept_classification, evidence_form="paraphrase" if any(reference.evidence_form == "paraphrase" for reference in candidate.concept_evidence) else "quotation", **feedback))
    if not understanding.ambiguous_identity:
        seen_concepts: set[str] = set()
        for extracted in validation.results:
            if not extracted.accepted:
                continue
            normalized_name = normalize_concept_term(extracted.canonical_name)
            if normalized_name in seen_concepts:
                continue
            seen_concepts.add(normalized_name)
            canonical = resolve_canonical_concept(db, extracted.canonical_name, extracted.aliases, extracted.definition, extracted.domain, extracted.mechanism, extracted.concept_verification, extracted.label_kind) if extracted.concept_classification == "established_concept" else None
            key = concept_feedback_key(extracted.canonical_name, extracted.concept_classification)
            feedback = feedback_state(item_id, key, extracted.canonical_name, db)
            problem_scope = {problem.problem_id: problem.spoiler_scope for problem in research_anchors}
            scope_order = {"premise_only": 0, "within_progress": 1, "full": 2}
            extracted_scope = max((problem_scope.get(problem_id, "premise_only") for problem_id in next((candidate.problem_ids for candidate in candidates.candidates if candidate.candidate_id == extracted.candidate_id), [])), key=lambda value: scope_order[value], default="premise_only")
            concept = ItemConcept(item_id=item_id, canonical_concept_id=canonical.id if canonical else None, name=extracted.canonical_name.strip(), canonical_name=extracted.canonical_name.strip(), category="mechanism" if extracted.level == "mechanism" else "general", origin="ai", explanation=extracted.definition, definition=extracted.definition, domain=extracted.domain, mechanism=extracted.mechanism, level=extracted.level, relationship=extracted.relationship_evidence, relationship_type=extracted.relationship_type, relationship_evidence=extracted.relationship_evidence, evidence_status="supported" if extracted.concept_verification == "verified" else "inferred", concept_verification=extracted.concept_verification, relationship_verification=extracted.relationship_verification, confidence=extracted.confidence, label_kind=extracted.label_kind, curiosity_hook=extracted.curiosity_hook, discovery_origin=extracted.discovery_origin, research_path=extracted.research_path, discovery_source_ids_json=json.dumps(extracted.discovery_source_ids), initial_validation_outcome=initial_evidence_outcomes.get(extracted.candidate_id), recovery_attempt_json=json.dumps(recovery_attempts.get(extracted.candidate_id, {})), final_validation_outcome="Accepted after final validation.", concept_classification=extracted.concept_classification, evidence_form="paraphrase" if any(reference.evidence_form == "paraphrase" for reference in extracted.concept_evidence) else "quotation", spoiler_scope=extracted_scope, **feedback)
            db.add(concept)
            db.flush()
            support_by_url: dict[str, set[str]] = {}
            for reference in extracted.concept_evidence:
                url = concept_source_urls.get(reference.source_id)
                if url:
                    support_by_url.setdefault(url, set()).add("concept")
            for source_id in extracted.relationship_source_ids:
                url = work_source_urls.get(source_id)
                if url:
                    support_by_url.setdefault(url, set()).add("relationship")
                    if url in saved_sources:
                        relationship_roles = set(json.loads(saved_sources[url].evidence_roles_json or "[]"))
                        relationship_roles.add("concept_relationship")
                        saved_sources[url].evidence_roles_json = json.dumps(sorted(relationship_roles))
            for source_id in extracted.discovery_source_ids:
                url = concept_source_urls.get(source_id)
                if url:
                    support_by_url.setdefault(url, set()).add("discovery")
            for url, support_kinds in support_by_url.items():
                if url in saved_sources:
                    support_kind = "both" if support_kinds == {"concept", "relationship"} else "_and_".join(sorted(support_kinds))
                    db.add(ConceptSource(concept_id=concept.id, source_id=saved_sources[url].id, support_kind=support_kind))
    db.commit()
    return enrichment_read(item_id, db)

@app.get("/api/exploration-invitations", response_model=list[ExplorationInvitationRead])
def list_exploration_invitations(
    item_id: int | None = None,
    status: Literal["queued", "dismissed"] | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(ExplorationInvitation)
    if item_id is not None:
        stmt = stmt.where(ExplorationInvitation.item_id == item_id)
    if status is not None:
        stmt = stmt.where(ExplorationInvitation.status == status)
    invitations = db.scalars(stmt.order_by(ExplorationInvitation.updated_at.desc(), ExplorationInvitation.id.desc())).all()
    return [exploration_invitation_read(invitation, db) for invitation in invitations]


@app.post("/api/concepts/{concept_id}/exploration", response_model=ExplorationInvitationRead)
def save_exploration_invitation(concept_id: int, db: Session = Depends(get_db)):
    concept = db.get(ItemConcept, concept_id)
    if concept is None or concept.superseded_at is not None:
        raise HTTPException(404, "Current discovery not found")
    if not concept.curiosity_hook or not concept.curiosity_hook.strip():
        raise HTTPException(422, "This discovery does not have a specific exploration invitation to save.")
    identity_filter = ExplorationInvitation.source_concept_id == concept.id
    if concept.feedback_key:
        identity_filter = or_(identity_filter, ExplorationInvitation.source_feedback_key == concept.feedback_key)
    invitation = db.scalar(select(ExplorationInvitation).where(
        ExplorationInvitation.item_id == concept.item_id, identity_filter,
    ).order_by(ExplorationInvitation.id.desc()))
    now = datetime.now(timezone.utc)
    if invitation is None:
        invitation = ExplorationInvitation(
            item_id=concept.item_id, source_concept_id=concept.id,
            source_feedback_key=concept.feedback_key, source_concept_name=concept.name,
            question=concept.curiosity_hook.strip(), status="queued", updated_at=now,
        )
    else:
        invitation.source_concept_id = concept.id
        invitation.source_feedback_key = concept.feedback_key
        invitation.source_concept_name = concept.name
        invitation.question = concept.curiosity_hook.strip()
        invitation.status = "queued"
        invitation.updated_at = now
    db.add(invitation)
    db.commit()
    db.refresh(invitation)
    return exploration_invitation_read(invitation, db)


@app.patch("/api/exploration-invitations/{invitation_id}", response_model=ExplorationInvitationRead)
def update_exploration_invitation(invitation_id: int, data: ExplorationInvitationUpdate, db: Session = Depends(get_db)):
    invitation = db.get(ExplorationInvitation, invitation_id)
    if invitation is None:
        raise HTTPException(404, "Exploration invitation not found")
    invitation.status = data.status
    invitation.user_note = data.note
    invitation.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(invitation)
    return exploration_invitation_read(invitation, db)


@app.get("/api/sidequests", response_model=list[SidequestRead])
def list_sidequests(db: Session = Depends(get_db)):
    saved = db.scalars(select(Sidequest).order_by(Sidequest.created_at.desc(), Sidequest.id.desc())).all()
    return [sidequest_read(sidequest, db) for sidequest in saved]

@app.post("/api/sidequests/surprise", response_model=SidequestRead, status_code=201)
def surprise_me(db: Session = Depends(get_db), ai: OpenRouterClient = Depends(get_ai_client)):
    items = db.scalars(select(CuriosityItem).order_by(CuriosityItem.id)).all()
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.superseded_at.is_(None)).order_by(ItemConcept.item_id, ItemConcept.id)).all()
    by_item: dict[int, list[ItemConcept]] = {}
    for concept in concepts:
        by_item.setdefault(concept.item_id, []).append(concept)
    queued_invitations = db.scalars(select(ExplorationInvitation).where(ExplorationInvitation.status == "queued")).all()
    invitation_by_concept: dict[int, ExplorationInvitation] = {}
    for invitation in queued_invitations:
        active_concept = active_invitation_concept(invitation, db)
        if active_concept is not None:
            invitation_by_concept[active_concept.id] = invitation
    researched_ids = set(db.scalars(select(ItemEnrichment.item_id).where(ItemEnrichment.status == "researched")).all())
    eligible = [item for item in items if by_item.get(item.id) and (not researched_ids or item.id in researched_ids or item.kind in {Kind.concept.value, Kind.question.value, Kind.thought.value})]
    previous_terms = set()
    for saved_concepts in db.scalars(select(Sidequest.concepts_json)).all():
        for name in json.loads(saved_concepts or "[]"):
            previous_terms.add(normalize_concept_term(str(name)))

    def pair_factors(pair: tuple[CuriosityItem, CuriosityItem]) -> dict[str, int]:
        item_a, item_b = pair
        a_concepts, b_concepts = by_item[item_a.id], by_item[item_b.id]
        combined = [*a_concepts, *b_concepts]
        a_terms = {normalize_concept_term(concept.name) for concept in a_concepts}
        b_terms = {normalize_concept_term(concept.name) for concept in b_concepts}
        return {
            "interest": sum(1 for concept in combined if concept.interested),
            "love": sum(1 for concept in combined if concept.loved),
            "surprise": min(1, sum(1 for concept in combined if concept.surprised)),
            "curiosity_potential": min(2, sum(1 for concept in combined if concept.id in invitation_by_concept)),
            "mechanisms": sum(1 for concept in combined if concept.level == "mechanism" or concept.category == "mechanism"),
            "grounded_discoveries": sum(1 for concept in combined if concept.concept_classification in {"established_concept", "grounded_interpretation", "grounded_observation"}),
            "media_diversity": 1 if item_a.kind != item_b.kind else 0,
            "overlap_penalty": len(a_terms & b_terms),
            "repetition_penalty": sum(1 for term in a_terms | b_terms if term in previous_terms),
        }

    def pair_score(pair: tuple[CuriosityItem, CuriosityItem]) -> tuple[int, int, int]:
        factors = pair_factors(pair)
        # Personal feedback raises relevance but is deliberately capped by small
        # weights; mechanism, grounding, diversity, and repetition remain visible.
        score = (
            factors["interest"] * 4 + factors["love"] * 6 + factors["surprise"] * 2
            + factors["curiosity_potential"] * 3
            + factors["mechanisms"] * 2 + factors["grounded_discoveries"]
            + factors["media_diversity"] * 2 - factors["overlap_penalty"] * 2
            - factors["repetition_penalty"] * 2
        )
        item_a, item_b = pair
        return (score, -item_a.id, -item_b.id)

    pairs = sorted(combinations(eligible, 2), key=pair_score, reverse=True)
    selected_pairs = pairs[:5]
    exploration_pair = next((pair for pair in pairs if pair not in selected_pairs and not any((concept.interested or concept.loved or concept.surprised) for item in pair for concept in by_item[item.id])), None)
    if exploration_pair is not None:
        selected_pairs.append(exploration_pair)
    def item_score(item: CuriosityItem) -> tuple[int, int]:
        item_concepts = by_item[item.id]
        score = (
            (5 if item.kind == Kind.thought.value else 0)
            + min(2, sum(1 for concept in item_concepts if concept.id in invitation_by_concept)) * 4
            + sum(int(concept.interested) + 2 * int(concept.loved) + int(concept.surprised) for concept in item_concepts)
            + sum(1 for concept in item_concepts if concept.level == "mechanism" or concept.category == "mechanism")
        )
        return (score, -item.id)

    selected_singles = sorted(eligible, key=item_score, reverse=True)[:3]
    if not selected_pairs and not selected_singles:
        raise HTTPException(422, "Add a concept or discovery to at least one curiosity before discovering Sidequests.")

    def concept_payload(concept: ItemConcept) -> dict[str, object]:
        source_urls = [] if concept.origin == "user_kept" else list(db.scalars(select(ResearchSource.url).join(ConceptSource, ConceptSource.source_id == ResearchSource.id).where(ConceptSource.concept_id == concept.id)).all())
        generic = any(re.search(pattern, concept.name.casefold()) for pattern in (r"\bcoming of age\b", r"\bteam dynamics\b", r"\boppression\b", r"\bmilitary strateg"))
        invitation = invitation_by_concept.get(concept.id)
        return {"name": concept.name, "level": concept.level or "legacy", "classification": concept.concept_classification or "legacy", "explanation": concept.explanation, "mechanism": concept.mechanism, "relationship": concept.relationship, "evidence_status": concept.evidence_status or "unverified", "spoiler_scope": concept.spoiler_scope, "human_kept": concept.origin == "user_kept", "interested": concept.interested, "loved": concept.loved, "surprised": concept.surprised, "saved_exploration": {"question": invitation.question, "user_note": invitation.user_note} if invitation else None, "discovery_mode": concept.discovery_mode, "discovery_quality": json.loads(concept.discovery_quality_json or "{}"), "discovery_priority": "low" if generic else "high" if concept.level == "mechanism" else "normal", "sources": source_urls}

    def item_payload(item: CuriosityItem) -> dict[str, object]:
        return {**item_ai_payload(item), "id": item.id, "concepts": [concept_payload(concept) for concept in by_item[item.id]]}

    candidate_contexts: list[dict[str, object]] = []
    allowed_origins: set[frozenset[int]] = set()
    for item_a, item_b in selected_pairs:
        allowed_origins.add(frozenset((item_a.id, item_b.id)))
        candidate_contexts.append({"mode": "connection", "item_a": item_payload(item_a), "item_b": item_payload(item_b)})
    for item in selected_singles:
        allowed_origins.add(frozenset((item.id,)))
        candidate_contexts.append({"mode": "single_item_direction", "item_a": item_payload(item), "item_b": None})
    max_contexts = max(1, min(int(os.getenv("SIDEQUEST_MAX_CONTEXTS", "8")), 12))
    candidate_contexts = candidate_contexts[:max_contexts]
    allowed_origins = {frozenset((context["item_a"]["id"], context["item_b"]["id"])) if context["item_b"] else frozenset((context["item_a"]["id"],)) for context in candidate_contexts}  # type: ignore[index]

    def candidate_origin_ids(candidate) -> list[int]:
        return list(dict.fromkeys(candidate.origin_item_ids or [candidate.item_a_id, candidate.item_b_id]))

    try:
        generated = ai.generate_candidates(candidate_contexts)
        normalized_candidates = [candidate.model_copy(update={
            "origin_item_ids": candidate_origin_ids(candidate),
            "invitation": candidate.invitation or candidate.new_question,
        }) for candidate in generated.candidates]
        grounded_candidates = [candidate for candidate in normalized_candidates if frozenset(candidate_origin_ids(candidate)) in allowed_origins]
        if not grounded_candidates:
            raise AIProviderError("No grounded connection candidates were returned.")
        rejected_reasons = [reason for candidate in grounded_candidates if (reason := deterministic_rejection_reason(candidate))]
        valid_candidates = [candidate for candidate in grounded_candidates if deterministic_rejection_reason(candidate) is None]
        if not valid_candidates:
            summary = "; ".join(dict.fromkeys(rejected_reasons))
            raise HTTPException(422, f"No compelling connection found. {summary} Try enriching more varied curiosities or adding more specific concepts.")
        final = ai.critique_candidates([candidate.model_dump() for candidate in valid_candidates])
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    final_origin_ids = candidate_origin_ids(final)
    if not final.accepted or frozenset(final_origin_ids) not in allowed_origins:
        raise HTTPException(422, sidequest_rejection_detail(final.rejection_reason))
    if len(final.bridge.strip()) < 12 or len(final.where_it_breaks.strip()) < 20:
        raise HTTPException(422, sidequest_rejection_detail("The proposed connection was too broad or did not explain its limitations clearly enough."))
    selected_origin = frozenset(final_origin_ids)
    dedupe_material = "|".join((",".join(str(item_id) for item_id in sorted(selected_origin)), normalize_concept_term(final.bridge), normalize_concept_term(final.connection)))
    dedupe_key = hashlib.sha256(dedupe_material.encode()).hexdigest()
    repeated = db.scalars(select(Sidequest)).all()
    if any(
        saved.dedupe_key == dedupe_key or (
            frozenset(json.loads(saved.origin_item_ids_json or "[]") or (saved.item_a_id, saved.item_b_id)) == selected_origin
            and normalize_concept_term(saved.bridge) == normalize_concept_term(final.bridge)
        )
        for saved in repeated
    ):
        raise HTTPException(422, sidequest_rejection_detail("That direction has already been explored for these curiosities."))
    item_a_id = final_origin_ids[0]
    item_b_id = final_origin_ids[1] if len(final_origin_ids) > 1 else item_a_id
    sidequest = Sidequest(
        item_a_id=item_a_id, item_b_id=item_b_id, title=final.title.strip(),
        connection=final.connection.strip(), bridge=final.bridge.strip(), why_interesting=final.why_interesting.strip(),
        where_it_breaks=final.where_it_breaks.strip(), new_question=final.new_question.strip(),
        evidence_status="Exploratory and not externally verified.",
        concepts_json=json.dumps(final.concepts_used), evidence_json=json.dumps(final.evidence),
        origin_item_ids_json=json.dumps(final_origin_ids), discovery_kind=final.discovery_kind,
        invitation=(final.invitation or final.new_question).strip(), dedupe_key=dedupe_key,
    )
    db.add(sidequest)
    db.commit()
    db.refresh(sidequest)
    return sidequest_read(sidequest, db)

@app.patch("/api/items/{item_id}", response_model=ItemRead)
def update_item(item_id: int, data: ItemUpdate, db: Session = Depends(get_db)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    changes = data.model_dump(exclude_unset=True)
    for required in ("kind", "work_scope", "progress_status", "progress_unit", "spoiler_policy"):
        if required in changes and changes[required] is None:
            raise HTTPException(422, f"{required.replace('_', ' ').title()} cannot be null")
    if "title" in changes:
        if changes["title"] is None or not changes["title"].strip():
            raise HTTPException(422, "Title cannot be blank")
        changes["title"] = changes["title"].strip()
    current_progress = changes.get("progress_current", item.progress_current)
    total_progress = changes.get("progress_total", item.progress_total)
    progress_unit = changes.get("progress_unit", item.progress_unit)
    validate_progress_values(current_progress, total_progress, progress_unit.value if isinstance(progress_unit, ProgressUnit) else progress_unit)
    identity_changed = any(field in changes and changes[field] != getattr(item, field) for field in ("kind", "title", "creator", "release_year", "work_scope", "parent_work_title", "canonical_identity_id"))
    spoiler_scope_changed = any(field in changes and changes[field] != getattr(item, field) for field in ("progress_status", "progress_current", "progress_total", "progress_unit", "spoiler_policy"))
    if identity_changed:
        enrichment = db.get(ItemEnrichment, item_id)
        if enrichment is not None:
            retire_stale_identity_research(item_id, db)
            enrichment.status = "not_researched"
            enrichment.identity_status = "unresolved"
            enrichment.identity_key = None
            enrichment.identity_candidates_json = "[]"
            enrichment.identity_source_snapshot_json = "[]"
            enrichment.confirmed_at = None
            enrichment.resolved_work_scope = None
            enrichment.parent_work_title = None
            enrichment.canonical_identity_id = None
    elif spoiler_scope_changed:
        enrichment = db.get(ItemEnrichment, item_id)
        if enrichment is not None and enrichment.status == "researched":
            retire_stale_identity_research(item_id, db)
            enrichment.status = "not_researched"
    for field, value in changes.items():
        setattr(item, field, value)
    db.commit()
    db.refresh(item)
    return item

@app.delete("/api/items/{item_id}", status_code=204)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    db.execute(delete(ExplorationInvitation).where(ExplorationInvitation.item_id == item_id))
    db.execute(delete(ItemConcept).where(ItemConcept.item_id == item_id))
    db.execute(delete(Sidequest).where(or_(Sidequest.item_a_id == item_id, Sidequest.item_b_id == item_id)))
    db.delete(item)
    db.commit()
