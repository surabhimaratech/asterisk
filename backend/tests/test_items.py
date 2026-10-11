from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import CandidateBatch, ConceptExtraction, CritiquedConnection, ExtractedConcept, ConnectionCandidate, AIProviderError
from app.main import BACKEND_DIR, DATA_DIR, DB_PATH, Base, app, get_ai_client, get_db


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def create_item(client: TestClient, **overrides: str):
    payload = {
        "kind": "book",
        "title": "The Dawn of Everything",
        "note": "  Keep this spacing exactly.\nSecond line.  ",
        "url": "https://example.com/book",
        **overrides,
    }
    return client.post("/api/items", json=payload)


def test_default_database_path_is_stable() -> None:
    assert DB_PATH == BACKEND_DIR / "data" / "asterisk.db"
    assert DB_PATH.is_absolute()
    assert DATA_DIR.is_dir()


def test_create_and_read_item_preserves_user_note(client: TestClient) -> None:
    response = create_item(client)

    assert response.status_code == 201
    created = response.json()
    assert created["id"] > 0
    assert created["created_at"]
    assert created["note"] == "  Keep this spacing exactly.\nSecond line.  "

    listed = client.get("/api/items")
    assert listed.status_code == 200
    assert listed.json() == [created]


def test_search_and_filter_items(client: TestClient) -> None:
    create_item(client, title="Invisible Cities", note="Calvino and imagined places")
    create_item(client, kind="question", title="Why do murmuration patterns form?", note="Birds")

    search = client.get("/api/items", params={"q": "imagined"})
    assert search.status_code == 200
    assert [item["title"] for item in search.json()] == ["Invisible Cities"]

    filtered = client.get("/api/items", params={"kind": "question"})
    assert filtered.status_code == 200
    assert [item["kind"] for item in filtered.json()] == ["question"]


def test_update_item(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]

    response = client.patch(
        f"/api/items/{item_id}",
        json={"kind": "concept", "title": "Collective intelligence", "note": "Unedited user text"},
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "concept"
    assert response.json()["title"] == "Collective intelligence"
    assert response.json()["note"] == "Unedited user text"


def test_delete_item(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]

    assert client.delete(f"/api/items/{item_id}").status_code == 204
    assert client.get("/api/items").json() == []
    assert client.delete(f"/api/items/{item_id}").status_code == 404


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"kind": "book", "title": "   "}, 422),
        ({"kind": "unknown", "title": "Something"}, 422),
        ({"kind": "article", "title": "Something", "url": "not-a-url"}, 422),
    ],
)
def test_create_validation_failures(client: TestClient, payload: dict[str, str], expected_status: int) -> None:
    assert client.post("/api/items", json=payload).status_code == expected_status


def test_update_validation_and_missing_item(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]

    assert client.patch(f"/api/items/{item_id}", json={"title": " "}).status_code == 422
    assert client.patch(f"/api/items/{item_id}", json={"kind": None}).status_code == 422
    assert client.patch("/api/items/999", json={"title": "Missing"}).status_code == 404


class FakeAI:
    def __init__(self) -> None:
        self.extracted = [ExtractedConcept(name="Emergence", category="mechanism", explanation="Local interactions forming a pattern.")]
        self.reject = False
        self.fail = False

    def extract_concepts(self, _item: dict[str, object]) -> ConceptExtraction:
        if self.fail:
            raise AIProviderError("Provider unavailable")
        return ConceptExtraction(concepts=self.extracted)

    def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
        a = pairs[0]["item_a"]["id"]  # type: ignore[index]
        b = pairs[0]["item_b"]["id"]  # type: ignore[index]
        return CandidateBatch(candidates=[ConnectionCandidate(
            item_a_id=a, item_b_id=b, title="Coordination without a conductor",
            connection="Both entries explore how local choices can create organized collective behavior.",
            bridge="Decentralized coordination", why_interesting="It reframes control as an emergent property.",
            where_it_breaks="The biological system adapts through evolution while the engineered system follows designed rules.",
            new_question="When does local coordination become resilient rather than chaotic?",
        )])

    def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
        candidate = candidates[0]
        return CritiquedConnection(
            accepted=not self.reject, rejection_reason="Too superficial" if self.reject else "",
            **candidate, evidence_status="Exploratory and not externally verified.",
        )


def test_manual_concept_crud_and_reextraction_preserves_manual(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    manual = client.post(f"/api/items/{item_id}/concepts", json={"name": "My own lens", "category": "personal"})
    assert manual.status_code == 201
    assert manual.json()["origin"] == "user"

    fake = FakeAI()
    app.dependency_overrides[get_ai_client] = lambda: fake
    first = client.post(f"/api/items/{item_id}/concepts/extract")
    assert first.status_code == 200
    assert {concept["name"] for concept in first.json()} == {"My own lens", "Emergence"}

    fake.extracted = [ExtractedConcept(name="Feedback loops", category="mechanism", explanation="Outcomes influence later behavior.")]
    second = client.post(f"/api/items/{item_id}/concepts/extract")
    assert {concept["name"] for concept in second.json()} == {"My own lens", "Feedback loops"}
    assert client.delete(f"/api/concepts/{manual.json()['id']}").status_code == 204


def test_extraction_provider_error(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    fake = FakeAI()
    fake.fail = True
    app.dependency_overrides[get_ai_client] = lambda: fake
    response = client.post(f"/api/items/{item_id}/concepts/extract")
    assert response.status_code == 502
    assert "Provider unavailable" in response.json()["detail"]


def test_extraction_missing_credentials(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    item_id = create_item(client).json()["id"]
    response = client.post(f"/api/items/{item_id}/concepts/extract")
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_sidequest_generation_and_persistence(client: TestClient) -> None:
    book_id = create_item(client, title="Distributed systems").json()["id"]
    animal_id = create_item(client, kind="concept", title="Animal intelligence").json()["id"]
    for item_id, name in [(book_id, "Fault tolerance"), (animal_id, "Collective behavior")]:
        assert client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"}).status_code == 201
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    created = client.post("/api/sidequests/surprise")
    assert created.status_code == 201
    assert created.json()["evidence_status"] == "Exploratory and not externally verified."
    assert client.get("/api/sidequests").json() == [created.json()]


def test_sidequest_insufficient_data_and_rejected_candidate(client: TestClient) -> None:
    assert client.post("/api/sidequests/surprise").status_code == 422
    first = create_item(client).json()["id"]
    second = create_item(client, kind="question", title="Why do flocks turn together?").json()["id"]
    for item_id in (first, second):
        client.post(f"/api/items/{item_id}/concepts", json={"name": "Coordination", "category": "mechanism"})
    fake = FakeAI()
    fake.reject = True
    app.dependency_overrides[get_ai_client] = lambda: fake
    rejected = client.post("/api/sidequests/surprise")
    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "Too superficial"


def test_llm_response_models_reject_invalid_categories() -> None:
    with pytest.raises(ValueError):
        ExtractedConcept(name="Anything", category="unsupported", explanation="No")  # type: ignore[arg-type]
