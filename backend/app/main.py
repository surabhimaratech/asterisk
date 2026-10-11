from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Integer, String, Text, create_engine, select, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DB_PATH = Path(__file__).resolve().parents[1] / "curiosity.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
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
    thought = "thought"

class CuriosityItem(Base):
    __tablename__ = "curiosity_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    title: Mapped[str] = mapped_column(String(240), index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ItemCreate(BaseModel):
    kind: Kind
    title: str = Field(min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)

class ItemUpdate(BaseModel):
    kind: Kind | None = None
    title: str | None = Field(default=None, min_length=1, max_length=240)
    note: str | None = Field(default=None, max_length=10000)
    url: str | None = Field(default=None, max_length=2000)

class ItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: Kind
    title: str
    note: str | None
    url: str | None
    created_at: datetime

def get_db():
    with SessionLocal() as session:
        yield session

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Asterisk", description="There's more to the story.", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
    db.delete(item)
    db.commit()
