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

from .ai import AIConfigurationError, AIProviderError, EvidenceConceptCandidate, EntityCandidate, OpenRouterClient, get_ai_client
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

class CuriosityItem(Base):
    __tablename__ = "curiosity_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(240), index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    creator: Mapped[str | None] = mapped_column(String(300), nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
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
    interested: Mapped[bool] = mapped_column(default=False)
    interest_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    interested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

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
    interested: Mapped[bool] = mapped_column(default=False)
    interest_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    interested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemCreate(BaseModel):
    kind: Kind
    title: str = Field(min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)
    creator: str | None = Field(default=None, max_length=300)
    release_year: int | None = Field(default=None, ge=1000, le=3000)

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
    interested: bool
    interest_note: str | None
    interested_at: datetime | None
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
    interested: bool
    interest_note: str | None
    interested_at: datetime | None
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
    created_at: datetime

class ResearchSourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    url: str
    title: str
    publisher: str | None
    source_type: str
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
    identity_confidence: float | None
    identity_candidates: list[dict[str, object]] = Field(default_factory=list)
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
    additions = {
        "curiosity_items": {"creator": "VARCHAR(300)", "release_year": "INTEGER"},
        "item_concepts": {"canonical_concept_id": "INTEGER REFERENCES canonical_concepts(id)", "level": "VARCHAR(20)", "relationship": "TEXT", "evidence_status": "VARCHAR(40)", "confidence": "FLOAT", "canonical_name": "VARCHAR(160)", "definition": "TEXT", "domain": "VARCHAR(120)", "mechanism": "TEXT", "relationship_type": "VARCHAR(20)", "relationship_evidence": "TEXT", "concept_verification": "VARCHAR(30)", "relationship_verification": "VARCHAR(30)", "label_kind": "VARCHAR(30)", "curiosity_hook": "TEXT", "discovery_origin": "VARCHAR(30)", "research_path": "TEXT", "discovery_source_ids_json": "TEXT NOT NULL DEFAULT '[]'", "initial_validation_outcome": "TEXT", "recovery_attempt_json": "TEXT NOT NULL DEFAULT '{}'", "final_validation_outcome": "TEXT", "review_audit_id": "INTEGER REFERENCES concept_lead_audits(id)", "user_note": "TEXT", "concept_classification": "VARCHAR(40)", "evidence_form": "VARCHAR(20)", "interested": "BOOLEAN NOT NULL DEFAULT 0", "interest_note": "TEXT", "interested_at": "DATETIME", "superseded_at": "DATETIME"},
        "sidequests": {"concepts_json": "TEXT NOT NULL DEFAULT '[]'", "evidence_json": "TEXT NOT NULL DEFAULT '[]'"},
        "concept_sources": {"support_kind": "VARCHAR(20) NOT NULL DEFAULT 'relationship'"},
        "canonical_concepts": {"aliases_json": "TEXT NOT NULL DEFAULT '[]'", "label_kind": "VARCHAR(30) NOT NULL DEFAULT 'established_term'"},
        "concept_lead_audits": {"stage": "VARCHAR(30) NOT NULL DEFAULT 'concept_candidate'", "discovery_origin": "VARCHAR(30)", "discovery_source_ids_json": "TEXT NOT NULL DEFAULT '[]'", "research_path": "TEXT", "initial_validation_outcome": "TEXT", "recovery_attempt_json": "TEXT NOT NULL DEFAULT '{}'", "final_validation_outcome": "TEXT", "candidate_snapshot_json": "TEXT NOT NULL DEFAULT '{}'", "validation_snapshot_json": "TEXT NOT NULL DEFAULT '{}'", "source_snapshot_json": "TEXT NOT NULL DEFAULT '[]'", "automated_evidence_status": "VARCHAR(30) NOT NULL DEFAULT 'unverified'", "human_review_status": "VARCHAR(20) NOT NULL DEFAULT 'pending'", "human_review_note": "TEXT", "review_history_json": "TEXT NOT NULL DEFAULT '[]'", "kept_concept_id": "INTEGER", "last_evaluated_at": "DATETIME", "evidence_review_detail": "TEXT", "superseded_at": "DATETIME", "concept_classification": "VARCHAR(40)", "evidence_form": "VARCHAR(20)", "interested": "BOOLEAN NOT NULL DEFAULT 0", "interest_note": "TEXT", "interested_at": "DATETIME"},
        "item_enrichments": {"identity_status": "VARCHAR(30) NOT NULL DEFAULT 'unresolved'", "identity_key": "VARCHAR(300)", "resolved_media_type": "VARCHAR(40)", "identity_confidence": "FLOAT", "identity_candidates_json": "TEXT NOT NULL DEFAULT '[]'", "identity_source_snapshot_json": "TEXT NOT NULL DEFAULT '[]'", "confirmed_at": "DATETIME"},
        "research_sources": {"superseded_at": "DATETIME"},
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
    return SidequestRead(
        **{column.name: getattr(sidequest, column.name) for column in Sidequest.__table__.columns},
        item_a_title=item_a.title if item_a else "Deleted curiosity",
        item_b_title=item_b.title if item_b else "Deleted curiosity",
        concepts_used=json.loads(sidequest.concepts_json or "[]"),
        evidence=json.loads(sidequest.evidence_json or "[]"),
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
        interested=concept.interested, interest_note=concept.interest_note, interested_at=concept.interested_at,
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
        interested=lead.interested, interest_note=lead.interest_note, interested_at=lead.interested_at,
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
        identity_confidence=enrichment.identity_confidence if enrichment else None,
        identity_candidates=json.loads(enrichment.identity_candidates_json or "[]") if enrichment else [],
        confirmed_at=enrichment.confirmed_at if enrichment else None,
        researched_at=enrichment.researched_at if enrichment else None,
        sources=list(sources), concepts=[concept_read(concept, db) for concept in concepts],
        rejected_leads=[rejected_finding_read(lead, db) for lead in rejected],
    )


def normalize_concept_term(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


WORK_KINDS = {Kind.book.value, Kind.movie.value, Kind.tv_series.value, Kind.podcast.value, Kind.article.value}


def identity_key(candidate: EntityCandidate) -> str:
    parts = [candidate.media_type, normalize_concept_term(candidate.title), normalize_concept_term(candidate.creator), str(candidate.release_year or "")]
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


def interest_state_for_name(item_id: int, name: str, db: Session) -> tuple[bool, str | None, datetime | None]:
    """Carry an explicit interest signal across regenerated rows with the same concept label."""
    normalized = normalize_concept_term(name)
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id)).all()
    audits = db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id)).all()
    matches = [entry for entry in [*concepts, *audits] if normalize_concept_term(entry.name) == normalized]
    interested = [entry for entry in matches if entry.interested]
    if not interested:
        return False, None, None
    latest = max(interested, key=lambda entry: entry.interested_at or entry.created_at)
    return True, latest.interest_note, latest.interested_at


def synchronize_interest(item_id: int, name: str, interested: bool, note: str | None, at: datetime | None, db: Session) -> None:
    """Keep historical and current representations from reviving a reversed signal."""
    normalized = normalize_concept_term(name)
    concepts = db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id)).all()
    audits = db.scalars(select(ConceptLeadAudit).where(ConceptLeadAudit.item_id == item_id)).all()
    for entry in [*concepts, *audits]:
        if normalize_concept_term(entry.name) == normalized:
            entry.interested = interested
            entry.interest_note = note if interested else None
            entry.interested_at = at if interested else None


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
                evidence_form=lead.evidence_form,
                interested=lead.interested, interest_note=lead.interest_note, interested_at=lead.interested_at,
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
    synchronize_interest(concept.item_id, concept.name, data.interested, data.note, at, db)
    db.commit()
    db.refresh(concept)
    return concept_read(concept, db)


@app.patch("/api/rejected-findings/{finding_id}/interest", response_model=RejectedFindingRead)
def update_finding_interest(finding_id: int, data: InterestUpdate, db: Session = Depends(get_db)):
    lead = db.get(ConceptLeadAudit, finding_id)
    if lead is None or lead.accepted:
        raise HTTPException(404, "Rejected finding not found")
    at = datetime.now(timezone.utc) if data.interested else None
    synchronize_interest(lead.item_id, lead.name, data.interested, data.note, at, db)
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
        extraction = ai.extract_concepts({"kind": item.kind, "title": item.title, "note": item.note, "url": item.url})
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    now = datetime.now(timezone.utc)
    for previous in db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.origin == "ai", ItemConcept.superseded_at.is_(None))).all():
        previous.superseded_at = now
    for extracted in extraction.concepts:
        interested, interest_note, interested_at = interest_state_for_name(item_id, extracted.name, db)
        db.add(ItemConcept(item_id=item_id, name=extracted.name.strip(), category=extracted.category, origin="ai", explanation=extracted.explanation, interested=interested, interest_note=interest_note, interested_at=interested_at))
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
    payload = {"kind": item.kind, "title": item.title, "note": item.note, "url": item.url, "creator": item.creator, "release_year": item.release_year}
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
    changed = enrichment.identity_key != identity_key(candidate) and (enrichment.identity_key is not None or enrichment.status == "researched")
    if changed:
        retire_stale_identity_research(item_id, db)
    enrichment.identity_status = "confirmed"
    enrichment.identity_key = identity_key(candidate)
    enrichment.resolved_title = candidate.title
    enrichment.creator = candidate.creator or None
    enrichment.release_info = str(candidate.release_year) if candidate.release_year else None
    enrichment.resolved_media_type = candidate.media_type
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
    payload = {"kind": item.kind, "title": item.title, "note": item.note, "url": item.url, "creator": item.creator, "release_year": item.release_year}
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
                "media_type": enrichment.resolved_media_type,
            }
        if packet is None:
            packet = research.research(payload)
        work_source_payloads = [{"source_id": f"work-{index}", "title": source.title, "excerpt": source.excerpt[:2500]} for index, source in enumerate(packet.sources, 1)]
        work_source_urls = {payload["source_id"]: source.url for payload, source in zip(work_source_payloads, packet.sources)}
        understanding = ai.understand_work(payload, packet.synthesis, work_source_payloads)
        if item.kind in {Kind.concept.value, Kind.question.value, Kind.thought.value}:
            understanding = understanding.model_copy(update={"ambiguous_identity": False, "confirmation_question": "", "resolved_title": understanding.resolved_title or item.title, "identity_matches_sources": True})
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc

    if item.kind in WORK_KINDS and (understanding.ambiguous_identity or not understanding.identity_matches_sources):
        reason = understanding.identity_mismatch_reason or understanding.confirmation_question or "The retrieved sources do not match the confirmed work identity."
        raise HTTPException(502, f"{reason} Existing research was preserved; correct or resolve the identity before retrying.")

    try:
        problems = ai.identify_research_problems(understanding, work_source_payloads)
        work_source_ids = set(work_source_urls)
        problem_ids = [problem.problem_id for problem in problems.problems]
        if len(problem_ids) != len(set(problem_ids)):
            raise AIProviderError("Research problem generation returned duplicate identifiers. Existing research was preserved.")
        if problem_reason := next((reason for problem in problems.problems if (reason := research_problem_rejection(problem, work_source_ids, item.title))), None):
            raise AIProviderError(f"Research problem validation failed: {problem_reason} Existing research was preserved.")
        concept_packet = research.research_problems(payload, [problem.model_dump() for problem in problems.problems])
        all_concept_sources = list(concept_packet.sources)
        concept_source_payloads = [{"source_id": f"concept-{index}", "title": source.title, "excerpt": source.excerpt[:2200]} for index, source in enumerate(all_concept_sources, 1)]
        concept_source_urls = {source_payload["source_id"]: source.url for source_payload, source in zip(concept_source_payloads, all_concept_sources)}
        concept_source_text = {source_payload["source_id"]: f"{source.title} {source.excerpt}" for source_payload, source in zip(concept_source_payloads, all_concept_sources)}
        candidates = ai.extract_evidence_concepts(problems, concept_packet.synthesis, concept_source_payloads)
        recovery_attempts: dict[str, dict[str, object]] = {}
        initial_evidence_outcomes: dict[str, str] = {}
        recovered_candidates = []
        recovery_source_urls: set[str] = set()
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
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc

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
            saved = ResearchSource(item_id=item_id, url=source.url, title=source.title[:500], publisher=urlparse(source.url).netloc or None, source_type=source_type, excerpt=source.excerpt[:2500])
            db.add(saved)
        else:
            saved.title = source.title[:500]
            saved.publisher = urlparse(source.url).netloc or None
            saved.source_type = source_type
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
    for result in validation.results:
        if result.accepted and (quality_reason := concept_quality_rejection(result, work_source_ids, concept_source_text, concept_source_urls, item.title)):
            result.accepted = False
            result.rejection_reason = quality_reason
    results_by_candidate = {result.candidate_id: result for result in validation.results}
    candidates_by_problem = {problem.problem_id: [] for problem in problems.problems}
    for candidate in candidates.candidates:
        for problem_id in candidate.problem_ids:
            candidates_by_problem.setdefault(problem_id, []).append(candidate)
        result = results_by_candidate.get(candidate.candidate_id)
        motivating_detail = "\n".join(reference.excerpt for reference in candidate.concept_evidence)
        final_outcome = "Accepted after final validation." if result and result.accepted else (result.rejection_reason if result else "Relationship validation returned no result.")
        referenced_ids = set(candidate.discovery_source_ids) | {reference.source_id for reference in candidate.concept_evidence} | set(result.relationship_source_ids if result else [])
        source_payload_by_id = {source["source_id"]: source for source in [*work_source_payloads, *concept_source_payloads]}
        source_url_by_id = {**work_source_urls, **concept_source_urls}
        source_snapshots = [{**source_payload_by_id[source_id], "url": source_url_by_id.get(source_id, ""), "source_type": "work" if source_id in work_source_ids else "independent"} for source_id in referenced_ids if source_id in source_payload_by_id]
        evidence_issue = candidate_evidence_rejection(candidate, concept_source_text, concept_source_urls)
        interested, interest_note, interested_at = interest_state_for_name(item_id, candidate.canonical_name, db)
        db.add(ConceptLeadAudit(item_id=item_id, name=candidate.canonical_name, proposed_definition=candidate.definition, motivating_detail=motivating_detail, proposed_relationship=result.relationship_evidence if result else candidate.research_path, relationship_type=result.relationship_type if result else "unsupported", research_questions_json=json.dumps(candidate.problem_ids), accepted=bool(result and result.accepted), rejection_reason=None if result and result.accepted else final_outcome, stage="concept_candidate", discovery_origin=candidate.discovery_origin, discovery_source_ids_json=json.dumps(candidate.discovery_source_ids), research_path=candidate.research_path, initial_validation_outcome=initial_evidence_outcomes.get(candidate.candidate_id), recovery_attempt_json=json.dumps(recovery_attempts.get(candidate.candidate_id, {})), final_validation_outcome=final_outcome, candidate_snapshot_json=json.dumps(candidate.model_dump()), validation_snapshot_json=json.dumps(result.model_dump()) if result else "{}", source_snapshot_json=json.dumps(source_snapshots), automated_evidence_status="partially_supported" if evidence_issue is None else "unverified", evidence_review_detail=evidence_issue or "Independent concept evidence passed deterministic checks; the final automated outcome is preserved separately.", concept_classification=candidate.concept_classification, evidence_form="paraphrase" if any(reference.evidence_form == "paraphrase" for reference in candidate.concept_evidence) else "quotation", interested=interested, interest_note=interest_note, interested_at=interested_at))
    for problem in problems.problems:
        has_candidates = bool(candidates_by_problem.get(problem.problem_id))
        reason = None if has_candidates else next(iter(candidates.insufficient_evidence_reasons), "Independent research did not yield a source-grounded canonical concept for this problem.")
        db.add(ConceptLeadAudit(item_id=item_id, name=problem.intellectual_question, proposed_definition=problem.concrete_description, motivating_detail=problem.work_situation, proposed_relationship=", ".join(problem.relevant_domains), relationship_type="unsupported", research_questions_json=json.dumps(problem.research_queries), accepted=has_candidates, rejection_reason=None if has_candidates else f"Insufficient evidence: {reason}", stage="research_problem"))
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
            interested, interest_note, interested_at = interest_state_for_name(item_id, extracted.canonical_name, db)
            concept = ItemConcept(item_id=item_id, canonical_concept_id=canonical.id if canonical else None, name=extracted.canonical_name.strip(), canonical_name=extracted.canonical_name.strip(), category="mechanism" if extracted.level == "mechanism" else "general", origin="ai", explanation=extracted.definition, definition=extracted.definition, domain=extracted.domain, mechanism=extracted.mechanism, level=extracted.level, relationship=extracted.relationship_evidence, relationship_type=extracted.relationship_type, relationship_evidence=extracted.relationship_evidence, evidence_status="supported" if extracted.concept_verification == "verified" else "inferred", concept_verification=extracted.concept_verification, relationship_verification=extracted.relationship_verification, confidence=extracted.confidence, label_kind=extracted.label_kind, curiosity_hook=extracted.curiosity_hook, discovery_origin=extracted.discovery_origin, research_path=extracted.research_path, discovery_source_ids_json=json.dumps(extracted.discovery_source_ids), initial_validation_outcome=initial_evidence_outcomes.get(extracted.candidate_id), recovery_attempt_json=json.dumps(recovery_attempts.get(extracted.candidate_id, {})), final_validation_outcome="Accepted after final validation.", concept_classification=extracted.concept_classification, evidence_form="paraphrase" if any(reference.evidence_form == "paraphrase" for reference in extracted.concept_evidence) else "quotation", interested=interested, interest_note=interest_note, interested_at=interested_at)
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
    researched_ids = set(db.scalars(select(ItemEnrichment.item_id).where(ItemEnrichment.status == "researched")).all())
    eligible = [item for item in items if by_item.get(item.id) and (not researched_ids or item.id in researched_ids)]
    previous_terms = set()
    for saved_concepts in db.scalars(select(Sidequest.concepts_json)).all():
        for name in json.loads(saved_concepts or "[]"):
            previous_terms.add(normalize_concept_term(str(name)))

    def pair_score(pair: tuple[CuriosityItem, CuriosityItem]) -> tuple[float, int, int]:
        item_a, item_b = pair
        a_concepts, b_concepts = by_item[item_a.id], by_item[item_b.id]
        interest = sum(1 for concept in [*a_concepts, *b_concepts] if concept.interested)
        mechanisms = sum(1 for concept in [*a_concepts, *b_concepts] if concept.level == "mechanism" or concept.category == "mechanism")
        established = sum(1 for concept in [*a_concepts, *b_concepts] if concept.concept_classification in {"established_concept", "grounded_interpretation"})
        a_terms = {normalize_concept_term(concept.name) for concept in a_concepts}
        b_terms = {normalize_concept_term(concept.name) for concept in b_concepts}
        overlap_penalty = len(a_terms & b_terms) * 2
        repetition_penalty = sum(1 for term in a_terms | b_terms if term in previous_terms)
        diversity = 2 if item_a.kind != item_b.kind else 0
        score = interest * 8 + mechanisms * 2 + established + diversity - overlap_penalty - repetition_penalty
        return (score, -item_a.id, -item_b.id)

    pairs = sorted(combinations(eligible, 2), key=pair_score, reverse=True)
    if not pairs:
        raise HTTPException(422, "Add concepts to at least two curiosities before asking for a Sidequest.")
    pair_payloads = []
    allowed_pairs: set[frozenset[int]] = set()
    for item_a, item_b in pairs[:6]:
        allowed_pairs.add(frozenset((item_a.id, item_b.id)))
        def concept_payload(concept: ItemConcept) -> dict[str, object]:
            source_urls = [] if concept.origin == "user_kept" else list(db.scalars(select(ResearchSource.url).join(ConceptSource, ConceptSource.source_id == ResearchSource.id).where(ConceptSource.concept_id == concept.id)).all())
            generic = any(re.search(pattern, concept.name.casefold()) for pattern in (r"\bcoming of age\b", r"\bteam dynamics\b", r"\boppression\b", r"\bmilitary strateg"))
            return {"name": concept.name, "level": concept.level or "legacy", "classification": concept.concept_classification or "legacy", "explanation": concept.explanation, "relationship": concept.relationship, "evidence_status": concept.evidence_status or "unverified", "human_kept": concept.origin == "user_kept", "interested": concept.interested, "discovery_priority": "low" if generic else "high" if concept.level == "mechanism" else "normal", "sources": source_urls}
        pair_payloads.append({
            "item_a": {"id": item_a.id, "kind": item_a.kind, "title": item_a.title, "note": item_a.note, "concepts": [concept_payload(c) for c in by_item[item_a.id]]},
            "item_b": {"id": item_b.id, "kind": item_b.kind, "title": item_b.title, "note": item_b.note, "concepts": [concept_payload(c) for c in by_item[item_b.id]]},
        })
    try:
        generated = ai.generate_candidates(pair_payloads)
        grounded_candidates = [candidate for candidate in generated.candidates if frozenset((candidate.item_a_id, candidate.item_b_id)) in allowed_pairs]
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
    if not final.accepted or frozenset((final.item_a_id, final.item_b_id)) not in allowed_pairs:
        raise HTTPException(422, sidequest_rejection_detail(final.rejection_reason))
    if len(final.bridge.strip()) < 12 or len(final.where_it_breaks.strip()) < 20:
        raise HTTPException(422, sidequest_rejection_detail("The proposed connection was too broad or did not explain its limitations clearly enough."))
    selected_pair = frozenset((final.item_a_id, final.item_b_id))
    repeated = db.scalars(select(Sidequest)).all()
    if any(
        frozenset((saved.item_a_id, saved.item_b_id)) == selected_pair
        and normalize_concept_term(saved.bridge) == normalize_concept_term(final.bridge)
        for saved in repeated
    ):
        raise HTTPException(422, sidequest_rejection_detail("That conceptual bridge has already been explored for this pair."))
    sidequest = Sidequest(
        item_a_id=final.item_a_id, item_b_id=final.item_b_id, title=final.title.strip(),
        connection=final.connection.strip(), bridge=final.bridge.strip(), why_interesting=final.why_interesting.strip(),
        where_it_breaks=final.where_it_breaks.strip(), new_question=final.new_question.strip(),
        evidence_status="Exploratory and not externally verified.",
        concepts_json=json.dumps(final.concepts_used), evidence_json=json.dumps(final.evidence),
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
    if "kind" in changes and changes["kind"] is None:
        raise HTTPException(422, "Kind cannot be null")
    if "title" in changes:
        if changes["title"] is None or not changes["title"].strip():
            raise HTTPException(422, "Title cannot be blank")
        changes["title"] = changes["title"].strip()
    identity_changed = any(field in changes and changes[field] != getattr(item, field) for field in ("kind", "title", "creator", "release_year"))
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
    db.execute(delete(ItemConcept).where(ItemConcept.item_id == item_id))
    db.execute(delete(Sidequest).where(or_(Sidequest.item_a_id == item_id, Sidequest.item_b_id == item_id)))
    db.delete(item)
    db.commit()
