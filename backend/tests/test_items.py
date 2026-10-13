from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import CandidateBatch, ConceptExtraction, ConceptualLead, ConceptualLeadBatch, ConceptValidationBatch, CritiquedConnection, EnrichedConcept, EnrichmentAnalysis, ExtractedConcept, ConnectionCandidate, AIProviderError, ValidatedConcept, WorkUnderstanding
from app.main import BACKEND_DIR, DATA_DIR, DB_PATH, Base, app, get_ai_client, get_db
from app.quality import concept_quality_rejection, deterministic_rejection_reason
from app.research import ResearchPacket, RetrievedSource, get_research_provider


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
            connection="Local agents acting on incomplete signals can produce organized collective behavior without global oversight.",
            bridge="Decentralized coordination under incomplete information", why_interesting="It reframes control as an emergent property.",
            where_it_breaks="The biological system adapts through evolution while the engineered system follows designed rules.",
            new_question="When does local coordination become resilient rather than chaotic?",
        )])

    def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
        candidate = candidates[0]
        return CritiquedConnection(
            accepted=not self.reject, rejection_reason="Too superficial" if self.reject else "",
            **candidate, evidence_status="Exploratory and not externally verified.", concepts_used=["Decentralized coordination"], evidence=[],
        )

    def analyze_research(self, _item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EnrichmentAnalysis:
        return EnrichmentAnalysis(
            ambiguous_identity=False, confirmation_question="", resolved_title="The Dawn of Everything",
            creator="David Graeber and David Wengrow", release_info="2021",
            identity_summary="A work of history and anthropology.", concepts=[EnrichedConcept(
                name="Institutional experimentation", level="concept", explanation="Societies vary political arrangements.",
                relationship="The work surveys changes in social organization.", evidence_status="supported", confidence=0.84,
                source_urls=[sources[0]["url"]],
            )],
        )

    def understand_work(self, item: dict[str, object], _synthesis: str, _sources: list[dict[str, str]]) -> WorkUnderstanding:
        return WorkUnderstanding(ambiguous_identity=False, confirmation_question="", resolved_title=str(item["title"]), creator="Test creator", release_info="2021", identity_summary="A researched work.", events_and_problems=["A community changes its institutions."], phenomena_and_processes=["Coordination under constraints."], dilemmas_and_decisions=["Actors choose without complete information."], systems_and_causal_relations=["Local choices alter collective outcomes."])

    def generate_conceptual_leads(self, _understanding: WorkUnderstanding) -> ConceptualLeadBatch:
        return ConceptualLeadBatch(leads=[ConceptualLead(name="Institutional experimentation", proposed_definition="Testing alternate rules for collective life.", motivating_detail="The work compares changing arrangements.", proposed_relationship="The work illustrates institutional variation.", relationship_type="illustrated", research_questions=["Is this an established concept?"])])

    def validate_concepts(self, _understanding: WorkUnderstanding, leads: ConceptualLeadBatch, _work_sources: list[dict[str, str]], _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        return ConceptValidationBatch(results=[ValidatedConcept(lead_name=leads.leads[0].name, accepted=True, rejection_reason="", canonical_name="Institutional experimentation", level="concept", definition="Deliberate or emergent testing of institutional forms.", domain="political anthropology", mechanism="Rules change incentives and collective behavior.", relationship_type="illustrated", relationship_evidence="The work compares changes in social organization.", concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.84, concept_source_urls=[concept_sources[0]["url"]], relationship_source_urls=["https://example.com/publisher"])])


class FakeResearch:
    def research(self, _item: dict[str, object]) -> ResearchPacket:
        source = RetrievedSource(url="https://example.com/publisher", title="Publisher description", excerpt="A sourced description.")
        return ResearchPacket(synthesis="The publisher identifies the work and its scope.", sources=[source, source])

    def research_concepts(self, _item: dict[str, object], _leads: list[dict[str, object]]) -> ResearchPacket:
        source = RetrievedSource(url="https://example.edu/concept", title="Institutional experimentation reference", excerpt="Institutional experimentation tests alternate rules and arrangements.")
        return ResearchPacket(synthesis="The concept is independently established.", sources=[source])


class AmbiguousAI(FakeAI):
    def understand_work(self, _item: dict[str, object], _synthesis: str, _sources: list[dict[str, str]]) -> WorkUnderstanding:
        return WorkUnderstanding(ambiguous_identity=True, confirmation_question="Which edition or creator did you mean?", resolved_title="", creator="", release_info="", identity_summary="", events_and_problems=[], phenomena_and_processes=[], dilemmas_and_decisions=[], systems_and_causal_relations=[])


class RejectingAI(FakeAI):
    def validate_concepts(self, _understanding: WorkUnderstanding, leads: ConceptualLeadBatch, _work_sources: list[dict[str, str]], _concept_synthesis: str, _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        return ConceptValidationBatch(results=[ValidatedConcept(lead_name=leads.leads[0].name, accepted=False, rejection_reason="This is a broad literary theme without an independently defined mechanism.", canonical_name="", level="concept", definition="", domain="", mechanism="", relationship_type="interpretive", relationship_evidence="", concept_verification="unverified", relationship_verification="unverified", confidence=0.2, concept_source_urls=[], relationship_source_urls=[])])


class FailingConceptResearch(FakeResearch):
    def research_concepts(self, _item: dict[str, object], _leads: list[dict[str, object]]) -> ResearchPacket:
        raise AIProviderError("Targeted concept research failed")


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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    item_id = create_item(client).json()["id"]
    response = client.post(f"/api/items/{item_id}/concepts/extract")
    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]


def test_research_enrichment_persists_sources_and_preserves_manual_concepts(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    manual = client.post(f"/api/items/{item_id}/concepts", json={"name": "My lens", "category": "personal"}).json()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()

    first = client.post(f"/api/items/{item_id}/enrich")
    second = client.post(f"/api/items/{item_id}/enrich")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "researched"
    assert len(second.json()["sources"]) == 2
    assert {concept["name"] for concept in second.json()["concepts"]} == {"My lens", "Institutional experimentation"}
    enriched = next(concept for concept in second.json()["concepts"] if concept["origin"] == "ai")
    assert enriched["level"] == "concept"
    assert enriched["evidence_status"] == "supported"
    assert set(enriched["source_ids"]) == {source["id"] for source in second.json()["sources"]}
    assert enriched["concept_verification"] == "verified"
    assert enriched["relationship_verification"] == "supported_interpretation"
    assert any(concept["id"] == manual["id"] for concept in second.json()["concepts"])


def test_ambiguous_refresh_requests_confirmation_without_erasing_existing_research(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    assert client.post(f"/api/items/{item_id}/enrich").status_code == 200

    app.dependency_overrides[get_ai_client] = lambda: AmbiguousAI()
    ambiguous = client.post(f"/api/items/{item_id}/enrich")

    assert ambiguous.status_code == 200
    assert ambiguous.json()["status"] == "needs_confirmation"
    assert ambiguous.json()["confirmation_question"] == "Which edition or creator did you mean?"
    assert len(ambiguous.json()["sources"]) == 2
    assert any(concept["origin"] == "ai" for concept in ambiguous.json()["concepts"])


def test_general_subject_bypasses_work_identity_confirmation(client: TestClient) -> None:
    item_id = create_item(client, kind="concept", title="Distributed Systems").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: AmbiguousAI()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    assert response.json()["status"] == "researched"
    assert response.json()["resolved_title"] == "Distributed Systems"


def test_rejected_lead_reason_is_persisted(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: RejectingAI()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    assert response.json()["rejected_leads"][0]["name"] == "Institutional experimentation"
    assert "broad literary theme" in response.json()["rejected_leads"][0]["rejection_reason"]


def test_partial_concept_research_failure_preserves_previous_enrichment(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    before = client.post(f"/api/items/{item_id}/enrich").json()
    app.dependency_overrides[get_research_provider] = lambda: FailingConceptResearch()

    failed = client.post(f"/api/items/{item_id}/enrich")
    after = client.get(f"/api/items/{item_id}/enrichment").json()

    assert failed.status_code == 502
    assert "Targeted concept research failed" in failed.json()["detail"]
    assert [source["url"] for source in after["sources"]] == [source["url"] for source in before["sources"]]
    assert [concept["id"] for concept in after["concepts"]] == [concept["id"] for concept in before["concepts"]]


def test_canonical_concept_is_reused_across_items(client: TestClient) -> None:
    first = create_item(client).json()["id"]
    second = create_item(client, title="A second work").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()

    first_concepts = client.post(f"/api/items/{first}/enrich").json()["concepts"]
    second_concepts = client.post(f"/api/items/{second}/enrich").json()["concepts"]

    first_ai = next(concept for concept in first_concepts if concept["origin"] == "ai")
    second_ai = next(concept for concept in second_concepts if concept["origin"] == "ai")
    assert first_ai["canonical_concept_id"] == second_ai["canonical_concept_id"]


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


def test_sidequest_allows_two_items_of_the_same_type(client: TestClient) -> None:
    first = create_item(client, title="Distributed systems").json()["id"]
    second = create_item(client, title="The Overstory").json()["id"]
    for item_id, name in [(first, "Fault tolerance"), (second, "Ecological resilience")]:
        client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"})

    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert {created.json()["item_a_id"], created.json()["item_b_id"]} == {first, second}


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
    assert "none were strong enough" in rejected.json()["detail"]
    assert "Too superficial" in rejected.json()["detail"]
    assert "more specific notes or concepts" in rejected.json()["detail"]


def test_llm_response_models_reject_invalid_categories() -> None:
    with pytest.raises(ValueError):
        ExtractedConcept(name="Anything", category="unsupported", explanation="No")  # type: ignore[arg-type]


def test_llm_response_models_reject_extra_fields_and_excess_concepts() -> None:
    with pytest.raises(ValueError):
        ExtractedConcept.model_validate({
            "name": "Emergence", "category": "mechanism", "explanation": "A pattern forms.", "confidence": 0.9,
        })

    concepts = [
        {"name": f"Concept {index}", "category": "general", "explanation": "A careful inference."}
        for index in range(7)
    ]
    with pytest.raises(ValueError):
        ConceptExtraction.model_validate({"concepts": concepts})


@pytest.mark.parametrize("connection", [
    "Both works explore extraterrestrial intelligence and aliens.",
    "Both entries discuss science, but from different formats and perspectives.",
])
def test_quality_gate_rejects_obvious_comparisons(connection: str) -> None:
    candidate = ConnectionCandidate(
        item_a_id=1, item_b_id=2, title="A familiar comparison", connection=connection,
        bridge="A broad thematic similarity without a distinct causal mechanism",
        why_interesting="They appear related.",
        where_it_breaks="Their contexts and claims differ substantially in ways the analogy does not explain.",
        new_question="What else is similar?",
    )
    assert deterministic_rejection_reason(candidate) is not None


def test_concept_quality_gate_separates_concept_and_relationship_evidence() -> None:
    broad = ValidatedConcept(
        lead_name="Human curiosity", accepted=True, rejection_reason="", canonical_name="Human Curiosity",
        level="concept", definition="Explores curiosity in a fictional crew.", domain="Literature",
        mechanism="The characters seek information when they encounter something unfamiliar.",
        relationship_type="explicit", relationship_evidence="Characters investigate the object.",
        concept_verification="verified", relationship_verification="verified", confidence=0.9,
        concept_source_urls=["https://work.example/review"], relationship_source_urls=["https://work.example/review"],
    )
    assert "broad theme" in (concept_quality_rejection(broad, {"https://work.example/review"}, {"https://concept.example/reference"}) or "")

    unsupported = broad.model_copy(update={"canonical_name": "Inference under uncertainty", "definition": "A decision process that updates beliefs from incomplete evidence."})
    assert "independent concept-research source" in (concept_quality_rejection(unsupported, {"https://work.example/review"}, {"https://concept.example/reference"}) or "")

    invented = unsupported.model_copy(update={"concept_source_urls": ["https://concept.example/reference"], "relationship_source_urls": ["https://work.example/review"]})
    reason = concept_quality_rejection(invented, {"https://work.example/review"}, {"https://concept.example/reference"}, {"https://concept.example/reference": "Bayesian reasoning and decision theory"})
    assert "lack substantive retrieved content" in (reason or "")

    inaccessible = invented.model_copy(update={"canonical_name": "Agency detection"})
    reason = concept_quality_rejection(inaccessible, {"https://work.example/review"}, {"https://concept.example/reference"}, {"https://concept.example/reference": "Agency detection. Client Challenge: a required part of this site couldn't load. Please enable JavaScript to continue viewing this page."})
    assert "inaccessible" in (reason or "")

    theme_summary = invented.model_copy(update={"canonical_name": "Bureaucratic inertia", "concept_source_urls": ["https://www.litcharts.com/themes/bureaucracy"]})
    reason = concept_quality_rejection(theme_summary, {"https://work.example/review"}, {"https://www.litcharts.com/themes/bureaucracy"}, {"https://www.litcharts.com/themes/bureaucracy": "Bureaucratic inertia describes how established procedures and organizational interests resist institutional change over long periods."})
    assert "low-value theme summaries" in (reason or "")

    supported = invented.model_copy(update={"canonical_name": "Inference under uncertainty"})
    source_text = "Inference under uncertainty is a decision process that updates beliefs from incomplete evidence and explicitly represents competing explanations."
    assert concept_quality_rejection(supported, {"https://work.example/review"}, {"https://concept.example/reference"}, {"https://concept.example/reference": source_text}) is None


def test_sidequest_reports_when_all_generated_candidates_are_shallow(client: TestClient) -> None:
    first = create_item(client, title="First contact novel").json()["id"]
    second = create_item(client, title="Another science story").json()["id"]
    for item_id in (first, second):
        client.post(f"/api/items/{item_id}/concepts", json={"name": "Aliens", "category": "general"})

    class ShallowAI(FakeAI):
        def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
            a = pairs[0]["item_a"]["id"]  # type: ignore[index]
            b = pairs[0]["item_b"]["id"]  # type: ignore[index]
            return CandidateBatch(candidates=[ConnectionCandidate(
                item_a_id=a, item_b_id=b, title="Aliens in two stories",
                connection="Both works explore extraterrestrial intelligence and aliens.",
                bridge="A broad thematic similarity without a distinct causal mechanism",
                why_interesting="They share a topic.",
                where_it_breaks="Their contexts and claims differ substantially in ways the analogy does not explain.",
                new_question="What else is similar?",
            )])

    app.dependency_overrides[get_ai_client] = lambda: ShallowAI()
    response = client.post("/api/sidequests/surprise")
    assert response.status_code == 422
    assert "No compelling connection found" in response.json()["detail"]
    assert "broad shared topic" in response.json()["detail"]
