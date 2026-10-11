from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from pathlib import Path
import os
import shutil

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, delete, select, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .ai import AIConfigurationError, AIProviderError, OpenAIClient, get_ai_client

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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemConcept(Base):
    __tablename__ = "item_concepts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("curiosity_items.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(20))
    origin: Mapped[str] = mapped_column(String(20))
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemCreate(BaseModel):
    kind: Kind
    title: str = Field(min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        return normalize_url(value)

class ItemUpdate(BaseModel):
    kind: Kind | None = None
    title: str | None = Field(default=None, min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)

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
    created_at: datetime

class ConceptCategory(str, Enum):
    personal = "personal"
    general = "general"
    mechanism = "mechanism"

class ConceptOrigin(str, Enum):
    ai = "ai"
    user = "user"

class ConceptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: ConceptCategory
    explanation: str | None = Field(default=None, max_length=600)

class ConceptRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    item_id: int
    name: str
    category: ConceptCategory
    origin: ConceptOrigin
    explanation: str | None
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
    created_at: datetime

def normalize_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return str(TypeAdapter(AnyHttpUrl).validate_python(value.strip()))

def get_db():
    with SessionLocal() as session:
        yield session

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Asterisk", description="There's more to the story.", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def ai_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AIConfigurationError):
        return HTTPException(503, str(exc))
    return HTTPException(502, str(exc))

def sidequest_read(sidequest: Sidequest, db: Session) -> SidequestRead:
    item_a = db.get(CuriosityItem, sidequest.item_a_id)
    item_b = db.get(CuriosityItem, sidequest.item_b_id)
    return SidequestRead(
        **{column.name: getattr(sidequest, column.name) for column in Sidequest.__table__.columns},
        item_a_title=item_a.title if item_a else "Deleted curiosity",
        item_b_title=item_b.title if item_b else "Deleted curiosity",
    )

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
    return db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id).order_by(ItemConcept.created_at, ItemConcept.id)).all()

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
    db.delete(concept)
    db.commit()

@app.post("/api/items/{item_id}/concepts/extract", response_model=list[ConceptRead])
def extract_concepts(item_id: int, db: Session = Depends(get_db), ai: OpenAIClient = Depends(get_ai_client)):
    item = db.get(CuriosityItem, item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    try:
        extraction = ai.extract_concepts({"kind": item.kind, "title": item.title, "note": item.note, "url": item.url})
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    db.execute(delete(ItemConcept).where(ItemConcept.item_id == item_id, ItemConcept.origin == "ai"))
    for extracted in extraction.concepts:
        db.add(ItemConcept(item_id=item_id, name=extracted.name.strip(), category=extracted.category, origin="ai", explanation=extracted.explanation))
    db.commit()
    return db.scalars(select(ItemConcept).where(ItemConcept.item_id == item_id).order_by(ItemConcept.created_at, ItemConcept.id)).all()

@app.get("/api/sidequests", response_model=list[SidequestRead])
def list_sidequests(db: Session = Depends(get_db)):
    saved = db.scalars(select(Sidequest).order_by(Sidequest.created_at.desc(), Sidequest.id.desc())).all()
    return [sidequest_read(sidequest, db) for sidequest in saved]

@app.post("/api/sidequests/surprise", response_model=SidequestRead, status_code=201)
def surprise_me(db: Session = Depends(get_db), ai: OpenAIClient = Depends(get_ai_client)):
    items = db.scalars(select(CuriosityItem).order_by(CuriosityItem.id)).all()
    concepts = db.scalars(select(ItemConcept).order_by(ItemConcept.item_id, ItemConcept.id)).all()
    by_item: dict[int, list[ItemConcept]] = {}
    for concept in concepts:
        by_item.setdefault(concept.item_id, []).append(concept)
    eligible = [item for item in items if by_item.get(item.id)]
    pairs = [(a, b) for a, b in combinations(eligible, 2) if a.kind != b.kind]
    if not pairs:
        raise HTTPException(422, "Add concepts to at least two curiosities from different types before asking for a Sidequest.")
    pair_payloads = []
    allowed_pairs: set[frozenset[int]] = set()
    for item_a, item_b in pairs[:6]:
        allowed_pairs.add(frozenset((item_a.id, item_b.id)))
        pair_payloads.append({
            "item_a": {"id": item_a.id, "kind": item_a.kind, "title": item_a.title, "note": item_a.note, "concepts": [c.name for c in by_item[item_a.id]]},
            "item_b": {"id": item_b.id, "kind": item_b.kind, "title": item_b.title, "note": item_b.note, "concepts": [c.name for c in by_item[item_b.id]]},
        })
    try:
        generated = ai.generate_candidates(pair_payloads)
        valid_candidates = [candidate for candidate in generated.candidates if frozenset((candidate.item_a_id, candidate.item_b_id)) in allowed_pairs]
        if not valid_candidates:
            raise AIProviderError("No grounded connection candidates were returned.")
        final = ai.critique_candidates([candidate.model_dump() for candidate in valid_candidates])
    except (AIConfigurationError, AIProviderError) as exc:
        raise ai_http_error(exc) from exc
    if not final.accepted or frozenset((final.item_a_id, final.item_b_id)) not in allowed_pairs:
        raise HTTPException(422, final.rejection_reason or "No candidate was strong enough for a Sidequest.")
    if len(final.bridge.strip()) < 12 or len(final.where_it_breaks.strip()) < 20:
        raise HTTPException(422, "The connection was too superficial to save as a Sidequest.")
    sidequest = Sidequest(
        item_a_id=final.item_a_id, item_b_id=final.item_b_id, title=final.title.strip(),
        connection=final.connection.strip(), bridge=final.bridge.strip(), why_interesting=final.why_interesting.strip(),
        where_it_breaks=final.where_it_breaks.strip(), new_question=final.new_question.strip(),
        evidence_status=final.evidence_status.strip(),
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
