from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import AIProviderError, CandidateBatch, ConceptExtraction, ConceptValidationBatch, CritiquedConnection, EnrichedConcept, EnrichmentAnalysis, EntityCandidate, EntityResolution, EvidenceConceptBatch, EvidenceConceptCandidate, EvidenceRecoveryResult, EvidenceReference, ExtractedConcept, ConnectionCandidate, ResearchProblem, ResearchProblemBatch, ValidatedConcept, WorkUnderstanding
from app.main import BACKEND_DIR, DATA_DIR, DB_PATH, Base, CanonicalConcept, app, get_ai_client, get_db, migrate_additive_columns, resolve_canonical_concept, select_diverse_anchors
from app.quality import candidate_evidence_rejection, concept_quality_rejection, deterministic_rejection_reason, rebind_exact_evidence, research_problem_rejection
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


def test_work_scope_progress_and_spoiler_preferences_persist_and_validate(client: TestClient) -> None:
    created = create_item(
        client, title="The Final Empire", work_scope="installment",
        parent_work_title="Mistborn", canonical_identity_id="openlibrary:OL123W",
        progress_status="in_progress", progress_current=2, progress_total=3,
        progress_unit="book", spoiler_policy="through_progress",
    )

    assert created.status_code == 201
    item = created.json()
    assert item["work_scope"] == "installment"
    assert item["parent_work_title"] == "Mistborn"
    assert item["canonical_identity_id"] == "openlibrary:OL123W"
    assert (item["progress_current"], item["progress_total"], item["progress_unit"]) == (2, 3, "book")
    assert item["spoiler_policy"] == "through_progress"
    assert client.patch(f"/api/items/{item['id']}", json={"progress_current": 4}).status_code == 422
    assert client.post("/api/items", json={"kind": "book", "title": "Too far", "progress_current": 101, "progress_total": 100, "progress_unit": "percent"}).status_code == 422


class FakeAI:
    def __init__(self) -> None:
        self.extracted = [ExtractedConcept(name="Emergence", category="mechanism", explanation="Local interactions forming a pattern.")]
        self.reject = False
        self.fail = False

    def extract_concepts(self, _item: dict[str, object]) -> ConceptExtraction:
        if self.fail:
            raise AIProviderError("Provider unavailable")
        return ConceptExtraction(concepts=self.extracted)

    def resolve_identity(self, item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
        return EntityResolution(status="confirmed", reason="Exact title and media type match.", candidates=[EntityCandidate(
            candidate_id="exact-work", title=str(item["title"]), creator="Test creator", release_year=2021,
            media_type=str(item["kind"]), description="The exact saved work.", confidence=0.98,
            source_ids=[sources[0]["source_id"]],
        )])

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

    def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]]) -> ResearchProblemBatch:
        return ResearchProblemBatch(problems=[ResearchProblem(problem_id="problem-1", concrete_description="A community changes its institutional rules.", work_situation="The work compares changing social arrangements.", intellectual_question="How do communities test alternative institutional forms?", relevant_domains=["political anthropology"], evidence_source_ids=[work_sources[0]["source_id"]], research_queries=["established term institutional forms testing collective rules"], anchor_kind="system", concise_label="Testing social arrangements", interesting_reason="The work treats political organization as something communities can deliberately vary rather than a fixed evolutionary stage.", distinctiveness="The comparison centers concrete institutional experiments instead of a generic theme of social change.", discovery_mode="deepen_fascination", external_research_warranted=True, discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "medium", "rationale": "A concrete institutional mechanism with cross-domain potential."})])

    def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(candidate_id="candidate-1", problem_ids=[problems.problems[0].problem_id], discovery_origin="work_grounded", discovery_source_ids=[concept_sources[0]["source_id"]], research_path="Research into how communities test alternate rules led to an independent account of institutional experimentation.", label_kind="established_term", canonical_name="Institutional experimentation", aliases=["institutional experiments"], definition="Deliberate or emergent testing of institutional forms.", domain="political anthropology", mechanism="Rules change incentives and collective behavior.", concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="Institutional experimentation tests alternate rules and arrangements in collective life.")], curiosity_hook="When do changes in collective rules become durable institutional experiments?", uncertainty="The term is used differently across disciplines.")], insufficient_evidence_reasons=[])

    def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, _work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        candidate = candidates.candidates[0]
        return ConceptValidationBatch(results=[ValidatedConcept(candidate_id=candidate.candidate_id, accepted=True, rejection_reason="", discovery_origin=candidate.discovery_origin, discovery_source_ids=candidate.discovery_source_ids, research_path=candidate.research_path, label_kind=candidate.label_kind, canonical_name=candidate.canonical_name, aliases=candidate.aliases, level="concept", definition=candidate.definition, concept_evidence=candidate.concept_evidence, domain=candidate.domain, mechanism=candidate.mechanism, relationship_type="illustrated", relationship_evidence="The work compares changes in social organization.", concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.84, relationship_source_ids=[_work_sources[0]["source_id"]], curiosity_hook=candidate.curiosity_hook)])


class FakeResearch:
    def research(self, _item: dict[str, object]) -> ResearchPacket:
        source = RetrievedSource(url="https://example.com/publisher", title="Publisher description", excerpt="A sourced description.")
        return ResearchPacket(synthesis="The publisher identifies the work and its scope.", sources=[source, source])

    def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
        source = RetrievedSource(url="https://example.edu/concept", title="Institutional experimentation reference", excerpt="Institutional experimentation tests alternate rules and arrangements in collective life.")
        return ResearchPacket(synthesis="The concept is independently established.", sources=[source])


class AmbiguousAI(FakeAI):
    def resolve_identity(self, item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
        return EntityResolution(status="ambiguous", reason="Which edition or creator did you mean?", candidates=[EntityCandidate(candidate_id="first", title=str(item["title"]), creator="Creator One", release_year=2020, media_type=str(item["kind"]), description="First plausible work.", confidence=0.75, source_ids=[sources[0]["source_id"]]), EntityCandidate(candidate_id="second", title=str(item["title"]), creator="Creator Two", release_year=2021, media_type=str(item["kind"]), description="Second plausible work.", confidence=0.73, source_ids=[sources[0]["source_id"]])])


class RejectingAI(FakeAI):
    def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, _work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        candidate = candidates.candidates[0]
        return ConceptValidationBatch(results=[ValidatedConcept(candidate_id=candidate.candidate_id, accepted=False, rejection_reason="This is a broad literary theme without an independently defined mechanism.", discovery_origin=candidate.discovery_origin, discovery_source_ids=candidate.discovery_source_ids, research_path=candidate.research_path, label_kind="descriptive_label", canonical_name="", aliases=[], level="concept", definition="", concept_evidence=[], domain="", mechanism="", relationship_type="unsupported", relationship_evidence="", concept_verification="unverified", relationship_verification="unverified", confidence=0.2, relationship_source_ids=[], curiosity_hook=None)])


class FabricatedEvidenceAI(RejectingAI):
    def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        candidate = super().extract_evidence_concepts(problems, _concept_synthesis, concept_sources).candidates[0]
        return EvidenceConceptBatch(candidates=[candidate.model_copy(update={"concept_evidence": [EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="This fabricated quotation never appeared in any retrieved source and cannot establish the claim.")]})], insufficient_evidence_reasons=[])


class MissingSourceAI(RejectingAI):
    def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        candidate = super().extract_evidence_concepts(problems, _concept_synthesis, concept_sources).candidates[0]
        return EvidenceConceptBatch(candidates=[candidate.model_copy(update={"concept_evidence": [EvidenceReference(source_id="concept-missing", excerpt="Institutional governance experiments compare alternative rules while communities observe how collective behavior changes.")]})], insufficient_evidence_reasons=[])


class NewEvidenceResearch(FakeResearch):
    def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
        source = RetrievedSource(url="https://example.edu/new-evidence", title="Institutional experimentation and governance", excerpt="Institutional experimentation tests alternate rules and arrangements in collective life. Institutional governance experiments compare alternative rules while communities observe how collective behavior changes.")
        return ResearchPacket(synthesis="New independent evidence is available.", sources=[source])


class FailingConceptResearch(FakeResearch):
    def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
        raise AIProviderError("Targeted concept research failed")


class InsufficientEvidenceAI(FakeAI):
    def extract_evidence_concepts(self, _problems: ResearchProblemBatch, _concept_synthesis: str, _concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        return EvidenceConceptBatch(candidates=[], insufficient_evidence_reasons=["Retrieved pages mention the problem but do not define an independent concept."])

    def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, _candidates: EvidenceConceptBatch, _work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        return ConceptValidationBatch(results=[])


class AliasAI(FakeAI):
    def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(candidate_id="candidate-alias", problem_ids=[problems.problems[0].problem_id], discovery_origin="work_grounded", discovery_source_ids=[concept_sources[0]["source_id"]], research_path="Research into alternate institutional forms led to an established account of institutional experimentation.", label_kind="established_term", canonical_name="Institutional experiments", aliases=["Institutional experimentation"], definition="Deliberate or emergent testing of institutional forms.", domain="political anthropology", mechanism="Rules change incentives and collective behavior.", concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="Institutional experimentation tests alternate rules and arrangements in collective life.")], curiosity_hook=None, uncertainty="Terminology varies by discipline.")], insufficient_evidence_reasons=[])


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
    assert {concept["name"] for concept in second.json()["concepts"]} == {"My lens", "Testing social arrangements", "Institutional experimentation"}
    observation = next(concept for concept in second.json()["concepts"] if concept["concept_classification"] == "grounded_observation")
    assert observation["evidence_status"] == "work_supported"
    assert observation["canonical_concept_id"] is None
    enriched = next(concept for concept in second.json()["concepts"] if concept["concept_classification"] == "established_concept")
    assert enriched["level"] == "concept"
    assert enriched["evidence_status"] == "supported"
    assert set(enriched["source_ids"]) == {source["id"] for source in second.json()["sources"]}
    assert {link["support_kind"] for link in enriched["source_support"]} == {"concept_and_discovery", "relationship"}
    assert enriched["concept_verification"] == "verified"
    assert enriched["relationship_verification"] == "supported_interpretation"
    assert enriched["label_kind"] == "established_term"
    assert enriched["curiosity_hook"] == "When do changes in collective rules become durable institutional experiments?"
    assert enriched["discovery_origin"] == "work_grounded"
    assert "Research into how communities" in enriched["research_path"]
    assert any(concept["id"] == manual["id"] for concept in second.json()["concepts"])


def test_confirmed_identity_is_reused_on_refresh(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    assert client.post(f"/api/items/{item_id}/enrich").status_code == 200

    app.dependency_overrides[get_ai_client] = lambda: AmbiguousAI()
    ambiguous = client.post(f"/api/items/{item_id}/enrich")

    assert ambiguous.status_code == 200
    assert ambiguous.json()["status"] == "researched"
    assert ambiguous.json()["identity_status"] == "confirmed"
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


def rejected_finding(client: TestClient, ai: FakeAI | None = None) -> tuple[int, int]:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: ai or RejectingAI()
    response = client.post(f"/api/items/{item_id}/enrich")
    assert response.status_code == 200
    finding = next(lead for lead in response.json()["rejected_leads"] if lead["stage"] == "concept_candidate")
    return item_id, finding["id"]


def test_user_can_keep_rejected_relationship_without_rewriting_automated_outcome(client: TestClient) -> None:
    item_id, finding_id = rejected_finding(client)
    first = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep", "note": "This question still interests me."})
    repeated = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep", "note": "Keep exploring this."})

    assert first.status_code == 200
    assert repeated.status_code == 200
    assert first.json()["rejection_reason"] == repeated.json()["rejection_reason"]
    assert "broad literary theme" in repeated.json()["final_validation_outcome"]
    assert repeated.json()["human_review_status"] == "kept"
    assert repeated.json()["automated_evidence_status"] == "partially_supported"
    assert first.json()["kept_concept_id"] == repeated.json()["kept_concept_id"]
    concepts = client.get(f"/api/items/{item_id}/concepts").json()
    kept = [concept for concept in concepts if concept["origin"] == "user_kept"]
    assert len(kept) == 1
    assert kept[0]["canonical_concept_id"] is None
    assert kept[0]["evidence_status"] == "user_kept_partially_supported"
    assert kept[0]["relationship_type"] == "unsupported"
    assert kept[0]["source_ids"] == []
    assert kept[0]["user_note"] == "Keep exploring this."


def test_review_decisions_can_be_reversed_without_losing_history(client: TestClient) -> None:
    item_id, finding_id = rejected_finding(client)
    kept = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep"}).json()
    dismissed = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "dismiss", "note": "Not useful right now."}).json()
    pending = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "pending"}).json()

    assert kept["kept_concept_id"] is not None
    assert dismissed["human_review_status"] == "dismissed"
    assert dismissed["kept_concept_id"] is None
    assert pending["human_review_status"] == "pending"
    assert [event["decision"] for event in pending["review_history"]] == ["keep", "dismiss", "pending"]
    assert not [concept for concept in client.get(f"/api/items/{item_id}/concepts").json() if concept["origin"] == "user_kept"]


def test_fabricated_quotation_stays_unverified_when_user_keeps_it(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCE_RECOVERY_MAX_CANDIDATES", "0")
    item_id, finding_id = rejected_finding(client, FabricatedEvidenceAI())
    reviewed = client.post(f"/api/rejected-findings/{finding_id}/reevaluate")
    kept = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep"})

    assert reviewed.status_code == 200
    assert reviewed.json()["automated_evidence_status"] == "unverified"
    assert "not a substantive verbatim" in reviewed.json()["evidence_review_detail"]
    concept = next(concept for concept in client.get(f"/api/items/{item_id}/concepts").json() if concept["origin"] == "user_kept")
    assert kept.json()["rejection_reason"]
    assert concept["evidence_status"] == "user_kept_unverified"


def test_missing_source_can_be_re_evaluated_against_available_evidence(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCE_RECOVERY_MAX_CANDIDATES", "0")
    item_id, finding_id = rejected_finding(client, MissingSourceAI())
    before = client.get("/api/rejected-findings", params={"item_id": item_id}).json()[0]
    app.dependency_overrides[get_research_provider] = lambda: NewEvidenceResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    assert client.post(f"/api/items/{item_id}/enrich").status_code == 200
    after = client.post(f"/api/rejected-findings/{finding_id}/reevaluate").json()

    assert before["automated_evidence_status"] == "unverified"
    assert after["automated_evidence_status"] == "partially_supported"
    assert "original relationship rejection" in after["evidence_review_detail"]


def test_grounded_observation_is_not_misrepresented_as_a_rejected_concept(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: InsufficientEvidenceAI()
    result = client.post(f"/api/items/{item_id}/enrich").json()
    assert any(concept["concept_classification"] == "grounded_observation" for concept in result["concepts"])
    assert not [finding for finding in result["rejected_leads"] if finding["stage"] == "work_anchor"]


def test_review_api_validates_decisions_and_note_length(client: TestClient) -> None:
    _item_id, finding_id = rejected_finding(client)
    assert client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "verify"}).status_code == 422
    assert client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep", "note": "x" * 2001}).status_code == 422


def test_insufficient_evidence_reports_research_problem_stage(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: InsufficientEvidenceAI()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    observations = [concept for concept in response.json()["concepts"] if concept["concept_classification"] == "grounded_observation"]
    assert len(observations) == 1
    assert not [concept for concept in response.json()["concepts"] if concept["concept_classification"] == "established_concept"]


def test_partial_concept_research_failure_still_preserves_grounded_observation(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    before = client.post(f"/api/items/{item_id}/enrich").json()
    app.dependency_overrides[get_research_provider] = lambda: FailingConceptResearch()

    failed = client.post(f"/api/items/{item_id}/enrich")
    after = client.get(f"/api/items/{item_id}/enrichment").json()

    assert failed.status_code == 200
    assert any(concept["concept_classification"] == "grounded_observation" for concept in after["concepts"])
    assert not [concept for concept in after["concepts"] if concept["concept_classification"] == "established_concept"]
    assert before["concepts"]


def test_canonical_concept_is_reused_across_items(client: TestClient) -> None:
    first = create_item(client).json()["id"]
    second = create_item(client, title="A second work").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()

    first_concepts = client.post(f"/api/items/{first}/enrich").json()["concepts"]
    second_concepts = client.post(f"/api/items/{second}/enrich").json()["concepts"]

    first_ai = next(concept for concept in first_concepts if concept["concept_classification"] == "established_concept")
    second_ai = next(concept for concept in second_concepts if concept["concept_classification"] == "established_concept")
    assert first_ai["canonical_concept_id"] == second_ai["canonical_concept_id"]


def test_verified_alias_reuses_canonical_without_string_similarity_merge(client: TestClient) -> None:
    first = create_item(client).json()["id"]
    second = create_item(client, title="A second work").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    first_ai = next(concept for concept in client.post(f"/api/items/{first}/enrich").json()["concepts"] if concept["concept_classification"] == "established_concept")

    app.dependency_overrides[get_ai_client] = lambda: AliasAI()
    second_ai = next(concept for concept in client.post(f"/api/items/{second}/enrich").json()["concepts"] if concept["concept_classification"] == "established_concept")

    assert first_ai["canonical_concept_id"] == second_ai["canonical_concept_id"]
    assert second_ai["name"] == "Institutional experiments"


def test_research_problems_drive_retrieval_and_source_roles_are_distinct(client: TestClient) -> None:
    captured: list[dict[str, object]] = []

    class TrackingResearch(FakeResearch):
        def research_problems(self, item: dict[str, object], problems: list[dict[str, object]]) -> ResearchPacket:
            captured.extend(problems)
            return super().research_problems(item, problems)

    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: TrackingResearch()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    assert captured[0]["intellectual_question"].startswith("How do communities")
    assert captured[0]["research_queries"] == ["established term institutional forms testing collective rules"]
    assert {source["source_type"] for source in response.json()["sources"]} == {"work", "concept_research"}


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


class NamedWorkAI(FakeAI):
    def resolve_identity(self, item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
        source_id = sources[0]["source_id"]
        if item["title"] == "Castle":
            candidates = [
                EntityCandidate(candidate_id="argentine-film", title="The Castle", creator="Martín Benchimol", release_year=2023, media_type="movie", description="An Argentine documentary film.", confidence=0.84, source_ids=[source_id]),
                EntityCandidate(candidate_id="abc-series", title="Castle", creator="Andrew W. Marlowe", release_year=2009, media_type="tv_series", description="The American crime comedy-drama series featuring Richard Castle and Kate Beckett.", confidence=0.98, source_ids=[source_id]),
            ]
        else:
            candidates = [
                EntityCandidate(candidate_id="original-series", title="Game of Thrones", creator="David Benioff and D. B. Weiss", release_year=2011, media_type="tv_series", description="The original HBO television series.", confidence=0.99, source_ids=[source_id]),
                EntityCandidate(candidate_id="prequel", title="Game of Thrones: Aegon's Conquest", creator="HBO", release_year=2027, media_type="tv_series", description="A separate prequel project.", confidence=0.65, source_ids=[source_id]),
            ]
        return EntityResolution(status="confirmed", reason="One exact title and media-type match.", candidates=candidates)


class WorkFirstDiscoveryAI(FakeAI):
    """Deterministic acceptance fixture: work observations require no paid research."""

    anchors_by_title = {
        "Castle": [
            ("relationship", "Mystery as courtship", "Castle and Beckett turn episodic investigations into an evolving contest of trust, attraction, and professional restraint."),
            ("narrative_structure", "Cases inside a longer arc", "Self-contained weekly cases supply closure while the unresolved conspiracy and central relationship accumulate across seasons."),
            ("tone_or_humor", "Humor as investigative friction", "Castle's playful speculation and Beckett's procedural skepticism make comic disagreement part of how competing explanations are tested."),
        ],
        "The Hunger Games": [
            ("system", "Spectacle as political control", "The arena turns punishment into compulsory entertainment, making spectatorship itself part of the regime's control system."),
            ("relationship", "Private care as public resistance", "Small acts of care become politically disruptive when an authoritarian spectacle demands visible rivalry and obedience."),
            ("worldbuilding", "An arena engineered to narrate power", "The designed environment controls resources, threats, and camera attention so that physical survival also produces a political story."),
        ],
        "A Psalm for the Wild-Built": [
            ("philosophical_question", "Need beyond usefulness", "Dex's search asks what a person needs when productivity, service, and competence do not produce a felt sense of fulfillment."),
            ("relationship", "Memory meets fresh perception", "A human carrying accumulated obligations meets a robot whose questions make ordinary needs appear strange and newly examinable."),
            ("worldbuilding", "Ecological restraint as infrastructure", "The setting treats limits on extraction as a durable social arrangement rather than a temporary response to crisis."),
            ("tone_or_humor", "Companionship through patient questions", "Gentle conversation and small misunderstandings let companionship emerge through attention instead of dramatic rescue or conquest."),
        ],
    }

    def resolve_identity(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
        if item["title"] == "Castle":
            return NamedWorkAI().resolve_identity(item, synthesis, sources)
        return super().resolve_identity(item, synthesis, sources)

    def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]]) -> ResearchProblemBatch:
        anchors = self.anchors_by_title[str(understanding.resolved_title)]
        return ResearchProblemBatch(problems=[ResearchProblem(
            problem_id=f"anchor-{index}", anchor_kind=kind, concise_label=label,
            concrete_description=detail, work_situation=detail,
            interesting_reason=f"This is worth following because {detail[0].lower() + detail[1:]}",
            distinctiveness=f"This anchor isolates the specific mechanism behind {label.lower()} rather than reducing the work to a broad theme.",
            intellectual_question=f"What changes when we treat {label.lower()} as an operating mechanism rather than background?",
            relevant_domains=["narrative design", "social systems"], evidence_source_ids=[work_sources[0]["source_id"]],
            research_queries=[], discovery_mode="deepen_fascination" if index % 2 else "reveal_overlooked_perspective",
            external_research_warranted=False,
            discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "high" if index % 2 else "medium", "rationale": "A specific work-grounded mechanism."},
            mechanism=f"The concrete constraint changes attention and choices: {detail}",
            curiosity_invitation=f"What becomes possible when this specific constraint changes how people interpret the situation in {label.lower()}?",
            appeal_dimensions=["clever"] if kind in {"narrative_structure", "tone_or_humor"} else ["thought_provoking"],
            pattern_scope="single_work",
        ) for index, (kind, label, detail) in enumerate(anchors, 1)])


class WorkOnlyResearch(FakeResearch):
    def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
        raise AssertionError("Work-supported observations must not trigger external concept research.")


@pytest.mark.parametrize(("title", "expected_creator"), [("Castle", "Andrew W. Marlowe"), ("Game of Thrones", "David Benioff and D. B. Weiss")])
def test_entity_resolution_prefers_exact_tv_series_and_rejects_namesakes(client: TestClient, title: str, expected_creator: str) -> None:
    item_id = create_item(client, kind="tv_series", title=title).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: NamedWorkAI()

    response = client.post(f"/api/items/{item_id}/identity/resolve")

    assert response.status_code == 200
    assert response.json()["identity_status"] == "confirmed"
    assert response.json()["resolved_title"] == title
    assert response.json()["creator"] == expected_creator


def test_series_scope_cannot_silently_resolve_to_first_installment(client: TestClient) -> None:
    class FirstInstallmentOnlyAI(FakeAI):
        def resolve_identity(self, item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
            return EntityResolution(status="confirmed", reason="Title resembles the first novel.", candidates=[EntityCandidate(
                candidate_id="study-in-scarlet", title="Sherlock Holmes", creator="Arthur Conan Doyle",
                release_year=1887, media_type="book", work_scope="individual",
                parent_work_title="Sherlock Holmes", canonical_identity_id="work:first-novel",
                description="The first Sherlock Holmes novel.", confidence=0.99,
                source_ids=[sources[0]["source_id"]],
            )])

    item_id = create_item(client, title="Sherlock Holmes", work_scope="series").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FirstInstallmentOnlyAI()

    result = client.post(f"/api/items/{item_id}/identity/resolve")

    assert result.status_code == 200
    assert result.json()["identity_status"] == "ambiguous"
    assert result.json()["resolved_title"] is None
    rejected_confirmation = client.post(f"/api/items/{item_id}/identity/confirm", json={"candidate_id": "study-in-scarlet"})
    assert rejected_confirmation.status_code == 422
    assert "work scope" in rejected_confirmation.json()["detail"]


def test_series_identity_scope_and_provenance_are_persisted(client: TestClient) -> None:
    class SeriesAI(FakeAI):
        def resolve_identity(self, item: dict[str, object], _synthesis: str, sources: list[dict[str, str]]) -> EntityResolution:
            return EntityResolution(status="confirmed", reason="Exact series match.", candidates=[EntityCandidate(
                candidate_id="sherlock-canon", title="Sherlock Holmes", creator="Arthur Conan Doyle",
                release_year=1887, media_type="book", work_scope="series",
                canonical_identity_id="wikidata:Q192837", description="The Sherlock Holmes canon.",
                confidence=0.99, source_ids=[sources[0]["source_id"]],
            )])

    item_id = create_item(client, title="Sherlock Holmes", work_scope="series").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: SeriesAI()

    result = client.post(f"/api/items/{item_id}/identity/resolve")

    assert result.status_code == 200
    body = result.json()
    assert body["identity_status"] == "confirmed"
    assert body["resolved_work_scope"] == "series"
    assert body["canonical_identity_id"] == "wikidata:Q192837"
    assert body["identity_sources"][0]["source_id"] == "identity-1"
    assert body["identity_sources"][0]["url"].startswith("https://")


@pytest.mark.parametrize(("title", "kind", "expected_labels"), [
    ("Castle", "tv_series", {"Mystery as courtship", "Cases inside a longer arc", "Humor as investigative friction"}),
    ("The Hunger Games", "book", {"Spectacle as political control", "Private care as public resistance", "An arena engineered to narrate power"}),
    ("A Psalm for the Wild-Built", "book", {"Need beyond usefulness", "Memory meets fresh perception", "Ecological restraint as infrastructure", "Companionship through patient questions"}),
])
def test_work_first_acceptance_examples_produce_specific_standalone_discoveries(
    client: TestClient, title: str, kind: str, expected_labels: set[str],
) -> None:
    item_id = create_item(client, kind=kind, title=title).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: WorkFirstDiscoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    if title == "Castle":
        assert response.json()["resolved_title"] == "Castle"
        assert response.json()["creator"] == "Andrew W. Marlowe"
        assert response.json()["release_info"] == "2009"
    discoveries = [concept for concept in response.json()["concepts"] if concept["concept_classification"] == "grounded_observation"]
    assert {concept["name"] for concept in discoveries} == expected_labels
    assert all(concept["canonical_concept_id"] is None for concept in discoveries)
    assert all(concept["evidence_status"] == "work_supported" for concept in discoveries)
    assert all(concept["discovery_quality"]["intellectual_substance"] == "high" for concept in discoveries)
    assert {concept["discovery_mode"] for concept in discoveries} == {"deepen_fascination", "reveal_overlooked_perspective"}
    assert all(concept["mechanism"] and concept["curiosity_hook"].endswith("?") for concept in discoveries)
    assert all(concept["appeal_dimensions"] for concept in discoveries)
    assert {concept["pattern_scope"] for concept in discoveries} == {"single_work"}


def test_anchor_and_research_limits_are_bounded_and_configurable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVERY_MAX_ANCHORS", "4")
    monkeypatch.setenv("DISCOVERY_MAX_RESEARCH_ANCHORS", "2")
    captured: list[dict[str, object]] = []

    class ManyAnchorAI(FakeAI):
        def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]]) -> ResearchProblemBatch:
            kinds = ["event", "relationship", "worldbuilding", "narrative_structure", "tone_or_humor", "motif"]
            return ResearchProblemBatch(problems=[ResearchProblem(
                problem_id=f"candidate-{index}", anchor_kind=kind, concise_label=f"Distinct mechanism {index}",
                concrete_description=f"A specific sequence {index} changes how characters coordinate under a clearly described constraint.",
                work_situation=f"The retrieved work account describes sequence {index} and its concrete consequences for later choices.",
                interesting_reason=f"Sequence {index} exposes a distinct causal pattern in how limited information changes coordinated action.",
                distinctiveness=f"This isolates causal pattern {index} rather than naming a broad topic shared by many works.",
                intellectual_question=f"How does causal pattern {index} reshape coordination under constraint?",
                relevant_domains=["systems", "narrative"], evidence_source_ids=[work_sources[0]["source_id"]],
                research_queries=[f"causal pattern {index} coordination under constraint"], external_research_warranted=True,
                discovery_mode="deepen_fascination" if index % 2 else "reveal_overlooked_perspective",
                discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "medium", "rationale": "Concrete and transferable."},
            ) for index, kind in enumerate(kinds, 1)])

    class TrackingResearch(FakeResearch):
        def research_problems(self, item: dict[str, object], problems: list[dict[str, object]]) -> ResearchPacket:
            captured.extend(problems)
            return super().research_problems(item, problems)

    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: ManyAnchorAI()
    app.dependency_overrides[get_research_provider] = lambda: TrackingResearch()
    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    assert len(captured) == 2
    assert len([concept for concept in response.json()["concepts"] if concept["concept_classification"] == "grounded_observation"]) == 4


def test_spoiler_boundary_rejects_out_of_scope_observations_without_saving_them(client: TestClient) -> None:
    captured: list[dict[str, object]] = []

    class CapturingResearch(WorkOnlyResearch):
        def research(self, item: dict[str, object]) -> ResearchPacket:
            captured.append(item)
            return super().research(item)

    class LateSpoilerAI(WorkFirstDiscoveryAI):
        def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]]) -> ResearchProblemBatch:
            batch = super().identify_research_problems(understanding, work_sources)
            return batch.model_copy(update={"problems": [batch.problems[0].model_copy(update={"spoiler_scope": "full"})]})

    item_id = create_item(
        client, title="The Hunger Games", progress_status="in_progress",
        progress_current=1, progress_total=3, progress_unit="book", spoiler_policy="through_progress",
    ).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: LateSpoilerAI()
    app.dependency_overrides[get_research_provider] = lambda: CapturingResearch()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 502
    assert "spoiler boundary" in response.json()["detail"]
    assert captured[0]["spoiler_ceiling"] == "within_progress"
    assert captured[0]["progress"] == {"status": "in_progress", "current": 1, "total": 3, "unit": "book"}
    assert client.get(f"/api/items/{item_id}/concepts").json() == []


def test_saved_research_sources_expose_evidence_roles(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    roles = {role for source in response.json()["sources"] for role in source["evidence_roles"]}
    assert {"work_identity", "work_detail", "external_concept", "concept_relationship"} <= roles


def test_love_and_surprise_survive_only_exact_observation_reruns(client: TestClient) -> None:
    ai = WorkFirstDiscoveryAI()
    item_id = create_item(client, title="The Hunger Games").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()
    first = client.post(f"/api/items/{item_id}/enrich").json()
    target = next(concept for concept in first["concepts"] if concept["name"] == "Spectacle as political control")
    assert client.patch(f"/api/concepts/{target['id']}/feedback", json={"signal": "love", "active": True, "note": "This stayed with me."}).status_code == 200
    assert client.patch(f"/api/concepts/{target['id']}/feedback", json={"signal": "surprise", "active": True}).status_code == 200

    rerun = client.post(f"/api/items/{item_id}/enrich").json()
    carried = next(concept for concept in rerun["concepts"] if concept["name"] == target["name"])
    assert carried["loved"] is True and carried["surprised"] is True
    assert carried["love_note"] == "This stayed with me."
    cleared = client.patch(f"/api/concepts/{carried['id']}/feedback", json={"signal": "love", "active": False})
    assert cleared.status_code == 200 and cleared.json()["loved"] is False
    restored = client.patch(f"/api/concepts/{carried['id']}/feedback", json={"signal": "love", "active": True})
    assert restored.status_code == 200 and restored.json()["loved"] is True

    original = ai.anchors_by_title["The Hunger Games"]
    ai.anchors_by_title = {**ai.anchors_by_title, "The Hunger Games": [
        (original[0][0], original[0][1], "A different underlying observation happens to share the old display label but describes unrelated material."),
        *original[1:],
    ]}
    changed = client.post(f"/api/items/{item_id}/enrich").json()
    unrelated = next(concept for concept in changed["concepts"] if concept["name"] == target["name"])
    assert unrelated["loved"] is False and unrelated["surprised"] is False


def test_curiosity_profile_uses_only_explicit_positive_feedback_and_authored_lenses(client: TestClient) -> None:
    item_id = create_item(client, title="Castle", kind="tv_series").json()["id"]
    manual = client.post(f"/api/items/{item_id}/concepts", json={"name": "Playful mysteries", "category": "personal"})
    assert manual.status_code == 201
    app.dependency_overrides[get_ai_client] = lambda: WorkFirstDiscoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()
    discoveries = client.post(f"/api/items/{item_id}/enrich").json()["concepts"]
    humor = next(concept for concept in discoveries if concept["name"] == "Humor as investigative friction")
    feedback = client.patch(f"/api/concepts/{humor['id']}/feedback", json={
        "signal": "love", "active": True, "note": "I love the technology angle, but humor matters too.",
    })
    assert feedback.status_code == 200

    profile = client.get("/api/discovery-profile")

    assert profile.status_code == 200
    body = profile.json()
    assert body["authored_lenses"] == ["Playful mysteries"]
    clever = next(entry for entry in body["hypotheses"] if entry["label"] == "clever")
    assert clever["signals"]["love"] == 1
    assert clever["notes"] == ["I love the technology angle, but humor matters too."]
    assert body["exploration_budget"] == 1


def test_profile_context_reaches_discovery_without_becoming_evidence(client: TestClient) -> None:
    first_id = create_item(client, title="Castle", kind="tv_series").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: WorkFirstDiscoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()
    first = client.post(f"/api/items/{first_id}/enrich").json()
    humor = next(concept for concept in first["concepts"] if concept["name"] == "Humor as investigative friction")
    assert client.patch(f"/api/concepts/{humor['id']}/feedback", json={"signal": "love", "active": True}).status_code == 200
    captured: list[dict[str, object]] = []

    class ProfileAwareAI(WorkFirstDiscoveryAI):
        def understand_work(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> WorkUnderstanding:
            captured.append(item)
            return super().understand_work(item, synthesis, sources)

    second_id = create_item(client, title="A Psalm for the Wild-Built").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: ProfileAwareAI()
    second = client.post(f"/api/items/{second_id}/enrich")

    assert second.status_code == 200
    assert captured and captured[0]["curiosity_profile"]["hypotheses"]
    # Work support remains the only evidence for grounded observations.
    assert all(concept["source_support"] for concept in second.json()["concepts"] if concept["concept_classification"] == "grounded_observation")


def test_manual_feedback_does_not_transfer_to_generated_observation_with_same_label(client: TestClient) -> None:
    item_id = create_item(client, title="The Hunger Games").json()["id"]
    manual = client.post(f"/api/items/{item_id}/concepts", json={"name": "Spectacle as political control", "category": "personal"}).json()
    assert client.patch(f"/api/concepts/{manual['id']}/feedback", json={"signal": "love", "active": True}).status_code == 200
    app.dependency_overrides[get_ai_client] = lambda: WorkFirstDiscoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()

    result = client.post(f"/api/items/{item_id}/enrich").json()

    observation = next(concept for concept in result["concepts"] if concept["concept_classification"] == "grounded_observation" and concept["name"] == manual["name"])
    assert observation["loved"] is False


def test_research_stops_when_sources_do_not_match_confirmed_identity(client: TestClient) -> None:
    class MismatchedSourcesAI(FakeAI):
        def understand_work(self, item: dict[str, object], synthesis: str, sources: list[dict[str, str]]) -> WorkUnderstanding:
            understanding = super().understand_work(item, synthesis, sources)
            return understanding.model_copy(update={"identity_matches_sources": False, "identity_mismatch_reason": "Sources concern a similarly named prequel."})

    item_id = create_item(client, kind="tv_series", title="Game of Thrones").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: MismatchedSourcesAI()
    response = client.post(f"/api/items/{item_id}/enrich")
    assert response.status_code == 502
    assert "similarly named prequel" in response.json()["detail"]
    assert client.get(f"/api/items/{item_id}/concepts").json() == []


@pytest.mark.parametrize(
    ("title", "label", "definition", "mechanism", "source_excerpt", "relationship"),
    [
        (
            "Eragon / Inheritance Cycle",
            "Language as Power",
            "A grounded interpretation of language as an instrument that constrains what actors can cause or command.",
            "Shared rules give utterances force: speakers must know the relevant forms, while meaning and consequences depend on how those forms are used.",
            "In many traditions, words do more than describe events: socially recognized utterances can authorize actions, impose obligations, and change what participants may do.",
            "The work portrays learned words as rule-bound instruments whose use can directly alter events while imposing costs and constraints on speakers.",
        ),
        (
            "Cytonic",
            "Identity Fragmentation",
            "A grounded interpretation of identity becoming divided across roles, memories, or incompatible frames of experience.",
            "When memories and roles cannot be integrated into one continuous self-model, people may experience competing perspectives as discontinuous parts of identity.",
            "Psychological accounts describe identity disturbance through discontinuity in a person's sense of self, agency, memory, and the integration of roles across contexts.",
            "The work depicts a character navigating discontinuous memories and roles that complicate a stable, integrated understanding of self.",
        ),
    ],
)
def test_grounded_interpretations_are_accepted_without_academic_label_match(
    client: TestClient,
    title: str,
    label: str,
    definition: str,
    mechanism: str,
    source_excerpt: str,
    relationship: str,
) -> None:
    class InterpretationResearch(FakeResearch):
        def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
            source = RetrievedSource(url="https://example.edu/grounded-interpretation", title="Independent mechanism reference", excerpt=source_excerpt)
            return ResearchPacket(synthesis="Independent research supports the underlying mechanism without naming the literary interpretation.", sources=[source])

    class InterpretationAI(FakeAI):
        def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
            return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(
                candidate_id="grounded-interpretation", problem_ids=[problems.problems[0].problem_id],
                discovery_origin="work_grounded", discovery_source_ids=[concept_sources[0]["source_id"]],
                research_path="Research into the work's concrete problem led to an independent account of the underlying mechanism and a cautious descriptive interpretation.",
                label_kind="descriptive_label", concept_classification="grounded_interpretation",
                canonical_name=label, aliases=[], definition=definition, domain="interpretive psychology",
                mechanism=mechanism,
                concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt=source_excerpt, evidence_form="paraphrase")],
                curiosity_hook=None, uncertainty="This descriptive label is an interpretation, not established academic terminology.",
            )], insufficient_evidence_reasons=[])

        def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
            candidate = candidates.candidates[0]
            return ConceptValidationBatch(results=[ValidatedConcept(
                candidate_id=candidate.candidate_id, accepted=True, rejection_reason="",
                discovery_origin=candidate.discovery_origin, discovery_source_ids=candidate.discovery_source_ids,
                research_path=candidate.research_path, label_kind=candidate.label_kind,
                concept_classification="grounded_interpretation", canonical_name=candidate.canonical_name,
                aliases=[], level="mechanism", definition=candidate.definition,
                concept_evidence=candidate.concept_evidence, domain=candidate.domain, mechanism=candidate.mechanism,
                relationship_type="interpretive", relationship_evidence=relationship,
                concept_verification="partially_verified", relationship_verification="supported_interpretation",
                confidence=0.78, mechanism_defensibility="defensible",
                relationship_source_ids=[work_sources[0]["source_id"]], curiosity_hook=None,
            )])

    item_id = create_item(client, kind="book", title=title).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: InterpretationResearch()
    app.dependency_overrides[get_ai_client] = lambda: InterpretationAI()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    concept = next(concept for concept in response.json()["concepts"] if concept["name"] == label)
    assert concept["name"] == label
    assert concept["concept_classification"] == "grounded_interpretation"
    assert concept["relationship_verification"] == "supported_interpretation"
    assert concept["canonical_concept_id"] is None


def test_ambiguous_identity_requires_choice_and_confirmation_persists(client: TestClient) -> None:
    item_id = create_item(client, kind="book", title="The Echo").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    ambiguous_ai = AmbiguousAI()
    app.dependency_overrides[get_ai_client] = lambda: ambiguous_ai

    unresolved = client.post(f"/api/items/{item_id}/enrich")
    assert unresolved.status_code == 200
    assert unresolved.json()["status"] == "needs_confirmation"
    assert len(unresolved.json()["identity_candidates"]) == 2

    confirmed = client.post(f"/api/items/{item_id}/identity/confirm", json={"candidate_id": "second"})
    assert confirmed.status_code == 200
    assert confirmed.json()["identity_status"] == "confirmed"
    assert confirmed.json()["creator"] == "Creator Two"

    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    researched = client.post(f"/api/items/{item_id}/enrich")
    assert researched.status_code == 200
    assert researched.json()["identity_status"] == "confirmed"
    assert researched.json()["creator"] == "Creator Two"


def test_identity_change_retires_ai_findings_but_preserves_user_content(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    manual_id = client.post(f"/api/items/{item_id}/concepts", json={"name": "My lens", "category": "personal"}).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    assert client.post(f"/api/items/{item_id}/enrich").status_code == 200

    changed = client.patch(f"/api/items/{item_id}", json={"creator": "A corrected creator"})

    assert changed.status_code == 200
    concepts = client.get(f"/api/items/{item_id}/concepts").json()
    assert [concept["id"] for concept in concepts] == [manual_id]
    enrichment = client.get(f"/api/items/{item_id}/enrichment").json()
    assert enrichment["identity_status"] == "unresolved"
    assert all(finding["superseded_at"] for finding in enrichment["rejected_leads"])


def test_progress_change_retires_ai_research_but_preserves_identity_and_user_content(client: TestClient) -> None:
    item_id = create_item(client, progress_status="planned", spoiler_policy="avoid").json()["id"]
    manual_id = client.post(f"/api/items/{item_id}/concepts", json={"name": "My own question", "category": "personal"}).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    researched = client.post(f"/api/items/{item_id}/enrich")
    assert researched.status_code == 200
    identity_key = researched.json()["identity_key"]
    assert identity_key

    changed = client.patch(f"/api/items/{item_id}", json={
        "progress_status": "in_progress", "progress_current": 1,
        "progress_total": 3, "progress_unit": "book", "spoiler_policy": "through_progress",
    })

    assert changed.status_code == 200
    concepts = client.get(f"/api/items/{item_id}/concepts").json()
    assert [concept["id"] for concept in concepts] == [manual_id]
    enrichment = client.get(f"/api/items/{item_id}/enrichment").json()
    assert enrichment["status"] == "not_researched"
    assert enrichment["identity_status"] == "confirmed"
    assert enrichment["identity_key"] == identity_key


def test_identity_reset_retires_user_kept_findings_from_current_discovery(client: TestClient) -> None:
    item_id, finding_id = rejected_finding(client)
    kept = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep"}).json()
    manual_id = client.post(f"/api/items/{item_id}/concepts", json={"name": "My durable lens", "category": "personal"}).json()["id"]
    assert kept["kept_concept_id"] is not None

    reset = client.delete(f"/api/items/{item_id}/identity")

    assert reset.status_code == 200
    assert [concept["id"] for concept in client.get(f"/api/items/{item_id}/concepts").json()] == [manual_id]
    historical = client.get("/api/rejected-findings", params={"item_id": item_id}).json()
    assert all(finding["superseded_at"] for finding in historical)


def test_movie_and_tv_series_are_distinct_api_types(client: TestClient) -> None:
    movie = create_item(client, kind="movie", title="Normal People: The Film").json()
    series = create_item(client, kind="tv_series", title="Normal People").json()
    assert movie["kind"] == "movie"
    assert series["kind"] == "tv_series"
    assert [item["id"] for item in client.get("/api/items", params={"kind": "tv_series"}).json()] == [series["id"]]


def test_additive_migration_preserves_ids_and_only_recategorizes_known_tv_titles(tmp_path) -> None:
    database = tmp_path / "legacy.db"
    migration_engine = create_engine(f"sqlite:///{database}")
    with migration_engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE curiosity_items (id INTEGER PRIMARY KEY, kind VARCHAR(20), title VARCHAR(240), note TEXT, url VARCHAR(2000), created_at DATETIME)")
        connection.exec_driver_sql("INSERT INTO curiosity_items (id, kind, title) VALUES (1, 'movie', 'Castle'), (2, 'movie', 'An Ambiguous Title'), (3, 'movie', 'Game of Thrones')")
        for statement in [
            "CREATE TABLE canonical_concepts (id INTEGER PRIMARY KEY, normalized_name VARCHAR(160), name VARCHAR(160), definition TEXT, domain VARCHAR(120), mechanism TEXT, verification_status VARCHAR(30), created_at DATETIME)",
            "CREATE TABLE item_concepts (id INTEGER PRIMARY KEY, item_id INTEGER, name VARCHAR(120), category VARCHAR(20), origin VARCHAR(20), explanation TEXT, created_at DATETIME)",
            "CREATE TABLE sidequests (id INTEGER PRIMARY KEY)", "CREATE TABLE concept_sources (id INTEGER PRIMARY KEY)", "CREATE TABLE research_sources (id INTEGER PRIMARY KEY, item_id INTEGER)",
            "CREATE TABLE concept_lead_audits (id INTEGER PRIMARY KEY, item_id INTEGER)", "CREATE TABLE item_enrichments (item_id INTEGER PRIMARY KEY, status VARCHAR(30), resolved_title VARCHAR(300), creator VARCHAR(300), release_info VARCHAR(200), identity_summary TEXT, confirmation_question TEXT, researched_at DATETIME)",
        ]:
            connection.exec_driver_sql(statement)
        connection.exec_driver_sql("INSERT INTO item_concepts (id, item_id, name, category, origin) VALUES (10, 1, 'Old AI finding', 'general', 'ai')")
        connection.exec_driver_sql("INSERT INTO research_sources (id, item_id) VALUES (20, 1)")
        connection.exec_driver_sql("INSERT INTO concept_lead_audits (id, item_id) VALUES (30, 1)")
        connection.exec_driver_sql("INSERT INTO item_enrichments (item_id, status) VALUES (1, 'researched')")
    migrate_additive_columns(migration_engine, f"sqlite:///{database}")
    with migration_engine.begin() as connection:
        rows = connection.exec_driver_sql("SELECT id, kind, title FROM curiosity_items ORDER BY id").fetchall()
        preserved = connection.exec_driver_sql("SELECT (SELECT count(*) FROM item_concepts WHERE id = 10 AND superseded_at IS NOT NULL), (SELECT count(*) FROM research_sources WHERE id = 20 AND superseded_at IS NOT NULL), (SELECT count(*) FROM concept_lead_audits WHERE id = 30 AND superseded_at IS NOT NULL), (SELECT status FROM item_enrichments WHERE item_id = 1), (SELECT loved FROM item_concepts WHERE id = 10), (SELECT surprised FROM item_concepts WHERE id = 10)").fetchone()
        p0_defaults = connection.exec_driver_sql("SELECT work_scope, progress_status, progress_unit, spoiler_policy FROM curiosity_items WHERE id = 2").fetchone()
        p0_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(curiosity_items)").fetchall()}
    assert rows == [(1, "tv_series", "Castle"), (2, "movie", "An Ambiguous Title"), (3, "tv_series", "Game of Thrones")]
    assert preserved == (1, 1, 1, "needs_confirmation", 0, 0)
    assert p0_defaults == ("unknown", "unknown", "installment", "avoid")
    assert {"work_scope", "parent_work_title", "canonical_identity_id", "progress_status", "progress_current", "progress_total", "progress_unit", "spoiler_policy"} <= p0_columns

    with migration_engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO curiosity_items (id, kind, title) VALUES (4, 'movie', 'Castle')")
    migrate_additive_columns(migration_engine, f"sqlite:///{database}")
    with migration_engine.begin() as connection:
        future_entry = connection.exec_driver_sql("SELECT kind FROM curiosity_items WHERE id = 4").scalar_one()
        migration_count = connection.exec_driver_sql("SELECT count(*) FROM schema_migrations WHERE version = '2026-10-13-known-tv-series'").scalar_one()
    assert future_entry == "movie"
    assert migration_count == 1


def test_cited_paraphrase_passes_without_exact_phrase_but_fake_quote_fails() -> None:
    text = {"concept-1": "Researchers describe how observers update possible explanations from incomplete evidence while preserving uncertainty about hidden causes."}
    base = EvidenceConceptCandidate(candidate_id="interpretation", problem_ids=["p1"], discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Independent research described provisional inference from incomplete observations and connected it to the work situation.", label_kind="descriptive_label", concept_classification="grounded_interpretation", canonical_name="Reasoning from incomplete traces", aliases=[], definition="A synthesis about provisional inference.", domain="reasoning", mechanism="Observers compare incomplete observations with competing explanations while retaining uncertainty.", concept_evidence=[EvidenceReference(source_id="concept-1", excerpt="Observers retain several possible explanations when evidence is incomplete and the hidden cause remains uncertain.", evidence_form="paraphrase")], curiosity_hook=None, uncertainty="A synthesis, not established terminology.")
    assert candidate_evidence_rejection(base, text, {"concept-1": "https://example.edu/evidence"}) is None
    fake_quote = base.model_copy(update={"concept_evidence": [base.concept_evidence[0].model_copy(update={"evidence_form": "quotation"})]})
    assert "verbatim" in (candidate_evidence_rejection(fake_quote, text, {"concept-1": "https://example.edu/evidence"}) or "")


def test_interest_is_independent_from_review_and_evidence_status(client: TestClient) -> None:
    item_id, finding_id = rejected_finding(client)
    before = client.get("/api/rejected-findings", params={"item_id": item_id}).json()[0]
    interested = client.patch(f"/api/rejected-findings/{finding_id}/interest", json={"interested": True, "note": "Follow this thread"})
    assert interested.status_code == 200
    assert interested.json()["interested"] is True
    assert interested.json()["human_review_status"] == before["human_review_status"] == "pending"
    assert interested.json()["automated_evidence_status"] == before["automated_evidence_status"]

    kept = client.patch(f"/api/rejected-findings/{finding_id}/review", json={"decision": "keep"})
    concept_id = kept.json()["kept_concept_id"]
    concept = next(concept for concept in client.get(f"/api/items/{item_id}/concepts").json() if concept["id"] == concept_id)
    assert concept["id"] == concept_id and concept["interested"] is True
    undone = client.patch(f"/api/concepts/{concept_id}/interest", json={"interested": False})
    assert undone.json()["interested"] is False
    finding = client.get("/api/rejected-findings", params={"item_id": item_id}).json()[0]
    assert finding["human_review_status"] == "kept"
    assert finding["interested"] is False


def test_interest_persists_across_research_reruns(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    first = client.post(f"/api/items/{item_id}/enrich").json()
    first_concept = next(concept for concept in first["concepts"] if concept["name"] == "Institutional experimentation")
    marked = client.patch(f"/api/concepts/{first_concept['id']}/interest", json={"interested": True, "note": "Keep following this mechanism."})
    assert marked.status_code == 200

    rerun = client.post(f"/api/items/{item_id}/enrich")

    assert rerun.status_code == 200
    current = rerun.json()["concepts"]
    current_concept = next(concept for concept in current if concept["name"] == "Institutional experimentation")
    assert current_concept["id"] != first_concept["id"]
    assert current_concept["interested"] is True
    assert current_concept["interest_note"] == "Keep following this mechanism."

    cleared = client.patch(f"/api/concepts/{current_concept['id']}/interest", json={"interested": False})
    assert cleared.status_code == 200
    rerun_after_reversal = client.post(f"/api/items/{item_id}/enrich")
    assert rerun_after_reversal.status_code == 200
    reversed_concept = next(concept for concept in rerun_after_reversal.json()["concepts"] if concept["name"] == "Institutional experimentation")
    assert reversed_concept["interested"] is False


def test_interest_persists_across_quick_extraction_reruns(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    first = client.post(f"/api/items/{item_id}/concepts/extract").json()[0]
    client.patch(f"/api/concepts/{first['id']}/interest", json={"interested": True, "note": "Worth revisiting."})

    rerun = client.post(f"/api/items/{item_id}/concepts/extract")

    assert rerun.status_code == 200
    current = rerun.json()[0]
    assert current["id"] != first["id"]
    assert current["interested"] is True
    assert current["interest_note"] == "Worth revisiting."


def test_interest_prioritizes_sidequest_pairs_without_changing_verification(client: TestClient) -> None:
    first = create_item(client, title="First").json()["id"]
    second = create_item(client, title="Second").json()["id"]
    third = create_item(client, title="Third").json()["id"]
    concept_ids = []
    for item_id, name in [(first, "One mechanism"), (second, "Two mechanism"), (third, "Three mechanism")]:
        concept_ids.append(client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"}).json()["id"])
    for concept_id in concept_ids[:2]:
        assert client.patch(f"/api/concepts/{concept_id}/interest", json={"interested": True}).status_code == 200

    class CapturingAI(FakeAI):
        first_pair: tuple[int, int] | None = None
        def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
            self.first_pair = (pairs[0]["item_a"]["id"], pairs[0]["item_b"]["id"])  # type: ignore[index]
            return super().generate_candidates(pairs)

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    assert client.post("/api/sidequests/surprise").status_code == 201
    assert set(ai.first_pair or ()) == {first, second}
    concepts = client.get(f"/api/items/{first}/concepts").json()
    assert concepts[0]["concept_verification"] is None


def test_love_and_surprise_shape_sidequests_without_overriding_verification_or_exploration(client: TestClient) -> None:
    item_ids = [create_item(client, title=f"Curiosity {index}").json()["id"] for index in range(1, 5)]
    concept_ids = [
        client.post(f"/api/items/{item_id}/concepts", json={"name": f"Mechanism {index}", "category": "mechanism"}).json()["id"]
        for index, item_id in enumerate(item_ids, 1)
    ]
    assert client.patch(f"/api/concepts/{concept_ids[0]}/feedback", json={"signal": "love", "active": True}).status_code == 200
    assert client.patch(f"/api/concepts/{concept_ids[1]}/feedback", json={"signal": "surprise", "active": True}).status_code == 200

    class CapturingAI(FakeAI):
        pairs: list[tuple[int, int]] = []

        def generate_candidates(self, pairs: list[dict[str, object]]) -> CandidateBatch:
            self.pairs = [(pair["item_a"]["id"], pair["item_b"]["id"]) for pair in pairs]  # type: ignore[index]
            return super().generate_candidates(pairs)

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert item_ids[0] in ai.pairs[0]
    assert any(set(pair) == {item_ids[2], item_ids[3]} for pair in ai.pairs)
    assert client.get(f"/api/items/{item_ids[0]}/concepts").json()[0]["concept_verification"] is None
    assert created.json()["evidence_status"] == "Exploratory and not externally verified."


def test_sidequest_rejects_repeated_bridge_and_forces_exploratory_status(client: TestClient) -> None:
    first = create_item(client, title="First").json()["id"]
    second = create_item(client, title="Second").json()["id"]
    for item_id, name in [(first, "Distributed coordination"), (second, "Collective sensing")]:
        client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"})

    class OverclaimingAI(FakeAI):
        def critique_candidates(self, candidates: list[dict[str, object]]) -> CritiquedConnection:
            result = super().critique_candidates(candidates)
            return result.model_copy(update={"evidence_status": "Verified scientific relationship."})

    app.dependency_overrides[get_ai_client] = lambda: OverclaimingAI()
    created = client.post("/api/sidequests/surprise")
    repeated = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert created.json()["evidence_status"] == "Exploratory and not externally verified."
    assert repeated.status_code == 422
    assert "already been explored" in repeated.json()["detail"]


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


def test_research_problem_requires_work_evidence_and_descriptive_queries() -> None:
    problem = ResearchProblem(problem_id="p1", concrete_description="Investigators infer a system's purpose from behavior.", work_situation="The crew observes an unfamiliar mechanism.", intellectual_question="How can function be inferred without specifications?", relevant_domains=["philosophy of science", "engineering"], evidence_source_ids=["work-1"], research_queries=["inferring system function from observed behavior without specifications"], anchor_kind="system", concise_label="Inferring purpose from behavior", interesting_reason="The investigators must reason about intended function using only repeated behavior and incomplete observations of the mechanism.", distinctiveness="The problem concerns inference without specifications, not a generic encounter with unknown technology.")
    assert research_problem_rejection(problem, {"work-1"}, "A Fictional Work") is None

    invented_source = problem.model_copy(update={"evidence_source_ids": ["work-altered"]})
    assert "not present" in (research_problem_rejection(invented_source, {"work-1"}, "A Fictional Work") or "")

    title_anchored = problem.model_copy(update={"research_queries": ["A Fictional Work abductive reasoning"]})
    assert "work title" in (research_problem_rejection(title_anchored, {"work-1"}, "A Fictional Work") or "")
    generic_invitation = problem.model_copy(update={"curiosity_invitation": "What can we learn from this theme?"})
    assert "generic theme" in (research_problem_rejection(generic_invitation, {"work-1"}, "A Fictional Work") or "")


def test_personalized_anchor_selection_preserves_an_unfamiliar_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCOVERY_MAX_ANCHORS", "3")
    anchors = []
    for index, label in enumerate(("Privacy system", "Security tradeoff", "Technology constraint", "Quiet companionship"), 1):
        anchors.append(ResearchProblem(
            problem_id=f"p{index}", concrete_description=f"A specific grounded situation involving {label.lower()} changes a consequential choice.",
            work_situation=f"The work source describes the concrete operation and consequence of {label.lower()}.",
            intellectual_question=f"What follows when {label.lower()} changes how the characters understand their available choices?",
            relevant_domains=["systems"], evidence_source_ids=["work-1"], research_queries=[],
            concise_label=label, interesting_reason=f"The detail makes {label.lower()} visible through a specific choice and its consequences.",
            distinctiveness=f"This is a concrete {label.lower()} mechanism rather than a broad genre theme.",
            external_research_warranted=False, mechanism=f"A constraint involving {label.lower()} changes information, incentives, and behavior.",
            curiosity_invitation=f"When can {label.lower()} produce an outcome that its participants did not intend?",
            appeal_dimensions=["thought_provoking"],
            discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "medium", "rationale": "Grounded and specific."},
        ))

    selected, _ = select_diverse_anchors(anchors, {"privacy", "security", "technology"})

    assert len(selected) == 3
    assert any(anchor.concise_label == "Quiet companionship" for anchor in selected)


def test_concept_quality_gate_separates_concept_and_relationship_evidence() -> None:
    citation = EvidenceReference(source_id="concept-1", excerpt="Inference under uncertainty is a decision process that updates beliefs from incomplete evidence.")
    broad = ValidatedConcept(
        candidate_id="candidate-1", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into incomplete evidence led to an independently sourced account of inference under uncertainty.", label_kind="established_term", canonical_name="Human Curiosity", aliases=[],
        level="concept", definition="Explores curiosity in a fictional crew.", concept_evidence=[citation], domain="Literature",
        mechanism="The characters seek information when they encounter something unfamiliar.",
        relationship_type="explicit", relationship_evidence="Characters investigate the object.",
        concept_verification="verified", relationship_verification="verified", confidence=0.9,
        relationship_source_ids=["work-1"], curiosity_hook=None,
    )
    concept_text = {"concept-1": "Inference under uncertainty is a decision process that updates beliefs from incomplete evidence. It explicitly represents competing explanations and revises conclusions."}
    concept_urls = {"concept-1": "https://concept.example/reference"}
    assert "broad theme" in (concept_quality_rejection(broad, {"work-1"}, concept_text, concept_urls) or "")

    unsupported = broad.model_copy(update={"canonical_name": "Inference under uncertainty", "definition": "A decision process that updates beliefs from incomplete evidence.", "concept_evidence": []})
    assert "independent concept-research source" in (concept_quality_rejection(unsupported, {"work-1"}, concept_text, concept_urls) or "")

    invented = unsupported.model_copy(update={"concept_evidence": [citation]})
    reason = concept_quality_rejection(invented, {"work-1"}, {"concept-1": "Bayesian reasoning and decision theory"}, concept_urls)
    assert "lack substantive retrieved content" in (reason or "")

    inaccessible = invented.model_copy(update={"canonical_name": "Agency detection"})
    reason = concept_quality_rejection(inaccessible, {"work-1"}, {"concept-1": "Agency detection. Client Challenge: a required part of this site couldn't load. Please enable JavaScript to continue viewing this page."}, concept_urls)
    assert "inaccessible" in (reason or "")

    theme_summary = invented.model_copy(update={"canonical_name": "Bureaucratic inertia", "concept_evidence": [EvidenceReference(source_id="concept-1", excerpt="Bureaucratic inertia describes how established procedures and organizational interests resist institutional change over long periods.")]})
    reason = concept_quality_rejection(theme_summary, {"work-1"}, {"concept-1": "Bureaucratic inertia describes how established procedures and organizational interests resist institutional change over long periods."}, {"concept-1": "https://www.litcharts.com/themes/bureaucracy"})
    assert "low-value theme summaries" in (reason or "")

    supported = invented.model_copy(update={"canonical_name": "Inference under uncertainty", "concept_evidence": [citation]})
    assert concept_quality_rejection(supported, {"work-1"}, concept_text, concept_urls) is None

    alias_supported = supported.model_copy(update={"canonical_name": "Bayesian inference", "aliases": ["Bayesian reasoning"], "concept_evidence": [EvidenceReference(source_id="concept-1", excerpt="Bayesian reasoning updates prior beliefs when new evidence becomes available.")]})
    alias_text = "Bayesian reasoning updates prior beliefs when new evidence becomes available. The framework represents uncertainty with probabilities."
    assert concept_quality_rejection(alias_supported, {"work-1"}, {"concept-1": alias_text}, concept_urls) is None

    paraphrased = supported.model_copy(update={"concept_evidence": [EvidenceReference(source_id="concept-1", excerpt="This paraphrase is not present in the retrieved source and cannot verify the definition.")]})
    reason = concept_quality_rejection(paraphrased, {"work-1"}, concept_text, concept_urls)
    assert "verbatim excerpt" in (reason or "")


def test_quality_gate_accepts_grounded_descriptive_label_without_exact_name_match() -> None:
    excerpt = "People often infer purpose and intention from observable behavior even when the underlying causes remain uncertain."
    candidate = ValidatedConcept(
        candidate_id="descriptive-1", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into observed behavior led to a source explaining how people infer possible goals from incomplete evidence.", label_kind="descriptive_label", concept_classification="grounded_interpretation",
        canonical_name="Inferring purpose from behavior", aliases=[], level="mechanism",
        definition="Reasoning from observable behavior to a possible purpose when direct design information is unavailable.",
        concept_evidence=[EvidenceReference(source_id="concept-1", excerpt=excerpt)], domain="reasoning",
        mechanism="Observers compare repeated behavior with possible goals while retaining uncertainty about hidden causes.",
        relationship_type="interpretive", relationship_evidence="Investigators observe an unfamiliar structure and propose possible functions.",
        concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.82,
        relationship_source_ids=["work-1"],
        curiosity_hook="When does repeated behavior justify inferring purpose despite uncertain causes?",
    )
    source_text = {"concept-1": f"Research overview. {excerpt} Competing explanations should remain open."}
    assert concept_quality_rejection(candidate, {"work-1"}, source_text, {"concept-1": "https://example.edu/reasoning"}, "Rendezvous with Rama") is None


def test_evidence_candidate_structured_response_requires_label_kind_references_and_hook_field() -> None:
    payload = {
        "candidate_id": "candidate-1", "problem_ids": ["problem-1"], "discovery_origin": "work_grounded", "discovery_source_ids": ["concept-1"], "research_path": "Research into observed behavior led to a source explaining provisional inference from incomplete evidence.", "label_kind": "descriptive_label",
        "canonical_name": "Inferring purpose from behavior", "aliases": [],
        "definition": "A provisional inference about purpose based on observed behavior.", "domain": "reasoning",
        "mechanism": "Repeated observations are compared with possible goals while alternative causes remain open.",
        "concept_evidence": [{"source_id": "concept-1", "excerpt": "Observers infer possible goals from repeated behavior while alternative causes remain open."}],
        "uncertainty": "The inference remains provisional.",
    }
    with pytest.raises(ValueError):
        EvidenceConceptCandidate.model_validate(payload)
    validated = EvidenceConceptCandidate.model_validate({**payload, "curiosity_hook": None})
    assert validated.label_kind == "descriptive_label"
    with pytest.raises(ValueError):
        EvidenceConceptCandidate.model_validate({**payload, "curiosity_hook": None, "concept_evidence": [{"source_id": "", "excerpt": "A valid-looking excerpt with enough words to pass superficial checks."}]})


@pytest.mark.parametrize("label", [
    "Human Understanding and Alien Complexity",
    "Challenges of Encountering Aliens",
    "Rendezvous with Rama's mystery",
])
def test_quality_gate_rejects_generic_or_work_specific_descriptive_labels(label: str) -> None:
    excerpt = "People form explanations from incomplete observations while uncertainty about hidden causes remains."
    candidate = ValidatedConcept(
        candidate_id="descriptive-2", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into incomplete observations led to a source discussing explanations formed under uncertainty.", label_kind="descriptive_label",
        canonical_name=label, aliases=[], level="concept", definition="A broad description of interpreting incomplete evidence.",
        concept_evidence=[EvidenceReference(source_id="concept-1", excerpt=excerpt)], domain="reasoning",
        mechanism="Observers compare incomplete observations with possible explanations while tracking uncertainty about hidden causes.",
        relationship_type="interpretive", relationship_evidence="Investigators interpret incomplete observations of the artifact.",
        concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.7,
        relationship_source_ids=["work-1"], curiosity_hook=None,
    )
    reason = concept_quality_rejection(candidate, {"work-1"}, {"concept-1": excerpt}, {"concept-1": "https://example.edu/reasoning"}, "Rendezvous with Rama")
    assert reason is not None


def test_quality_gate_rejects_altered_source_id_and_unsupported_hook() -> None:
    excerpt = "People form explanations from incomplete observations while uncertainty about hidden causes remains."
    candidate = ValidatedConcept(
        candidate_id="descriptive-3", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into incomplete observations led to a source discussing provisional explanations under uncertainty.", label_kind="descriptive_label", concept_classification="grounded_interpretation",
        canonical_name="Reasoning from incomplete traces", aliases=[], level="mechanism",
        definition="Drawing provisional explanations from incomplete observations.",
        concept_evidence=[EvidenceReference(source_id="concept-altered", excerpt=excerpt)], domain="reasoning",
        mechanism="Observers compare incomplete observations with several explanations and retain uncertainty about hidden causes.",
        relationship_type="illustrated", relationship_evidence="Investigators compare observations with possible explanations.",
        concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.8,
        relationship_source_ids=["work-1"], curiosity_hook="Could volcanic farming on Venus eliminate all uncertainty?",
    )
    source_text = {"concept-1": excerpt}
    assert "does not match" in (concept_quality_rejection(candidate, {"work-1"}, source_text, {"concept-1": "https://example.edu/reasoning"}) or "")
    candidate = candidate.model_copy(update={"concept_evidence": [EvidenceReference(source_id="concept-1", excerpt=excerpt)]})
    assert "not grounded" in (concept_quality_rejection(candidate, {"work-1"}, source_text, {"concept-1": "https://example.edu/reasoning"}) or "")


def test_exact_excerpt_can_be_rebound_but_unrelated_real_source_cannot_pass() -> None:
    excerpt = "Rotating habitats create apparent gravity through centripetal acceleration directed toward the outer surface."
    candidate = EvidenceConceptCandidate(
        candidate_id="gravity", problem_ids=["problem-1"], discovery_origin="work_grounded",
        discovery_source_ids=["concept-1"], research_path="Research into a rotating habitat led to the mechanics of apparent gravity in cylindrical structures.",
        label_kind="established_term", canonical_name="Rotational artificial gravity", aliases=["artificial gravity"],
        definition="Apparent gravity produced by rotation.", domain="physics",
        mechanism="Rotation produces centripetal acceleration that presses occupants toward a habitat's outer surface.",
        concept_evidence=[EvidenceReference(source_id="concept-wrong", excerpt=excerpt)], curiosity_hook=None, uncertainty="",
    )
    texts = {"concept-1": f"Artificial gravity reference. {excerpt}", "concept-wrong": "A policy paper about extraterrestrial messaging and consent."}
    rejection = candidate_evidence_rejection(candidate, texts) or ""
    assert "inaccessible, low-value" in rejection or "not a substantive verbatim" in rejection
    rebound, events = rebind_exact_evidence(candidate, texts)
    assert rebound.concept_evidence[0].source_id == "concept-1"
    assert events[0]["outcome"] == "exact_excerpt_rebound"
    assert candidate_evidence_rejection(rebound, texts) is None


def test_research_led_discovery_can_pass_without_claiming_direct_work_support() -> None:
    excerpt = "Messaging extraterrestrial intelligence is the deliberate transmission of signals intended for possible extraterrestrial recipients."
    result = ValidatedConcept(
        candidate_id="meti", accepted=True, rejection_reason="", discovery_origin="research_led",
        discovery_source_ids=["concept-1"],
        research_path="Researching ethical responses to unknown intelligence led to a source debating whether humanity should deliberately transmit messages to extraterrestrial recipients.",
        label_kind="established_term", canonical_name="Messaging extraterrestrial intelligence", aliases=["METI"],
        level="concept", definition="Deliberately transmitting signals intended for extraterrestrial intelligence.",
        concept_evidence=[EvidenceReference(source_id="concept-1", excerpt=excerpt)], domain="space policy",
        mechanism="A sender intentionally broadcasts an encoded signal despite uncertainty about possible recipients and consequences.",
        relationship_type="unsupported", relationship_evidence="No direct relationship to the work is claimed.",
        concept_verification="verified", relationship_verification="unverified", confidence=0.88,
        relationship_source_ids=[], curiosity_hook="Who gets to decide whether humanity announces its existence to the universe?",
    )
    text = {"concept-1": excerpt + " Governance debates concern consent and representation across humanity."}
    assert concept_quality_rejection(result, {"work-1"}, text, {"concept-1": "https://example.edu/meti"}) is None
    weak_path = result.model_copy(update={"research_path": "It appeared in research."})
    assert "discovery path" in (concept_quality_rejection(weak_path, {"work-1"}, text, {"concept-1": "https://example.edu/meti"}) or "")


def test_targeted_evidence_recovery_is_bounded_and_persisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCE_RECOVERY_MAX_CANDIDATES", "1")
    monkeypatch.setenv("EVIDENCE_RECOVERY_MAX_ESTIMATED_COST_USD", "0.02")
    item_id = create_item(client).json()["id"]

    class RecoveryResearch(FakeResearch):
        calls = 0

        def recover_candidate(self, _item: dict[str, object], _candidate: dict[str, object]) -> ResearchPacket:
            self.calls += 1
            source = RetrievedSource(url="https://example.edu/artificial-gravity", title="Rotational artificial gravity", excerpt="Rotational artificial gravity uses centripetal acceleration to press occupants toward the outer surface of a rotating habitat.")
            return ResearchPacket(synthesis="Rotation can produce apparent gravity.", sources=[source])

    class RecoveryAI(FakeAI):
        def extract_evidence_concepts(self, problems: ResearchProblemBatch, _synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
            return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(
                candidate_id="gravity", problem_ids=[problems.problems[0].problem_id], discovery_origin="work_grounded",
                discovery_source_ids=[concept_sources[0]["source_id"]], research_path="Research into an engineered environment raised the question of how rotation can produce apparent gravity.",
                label_kind="established_term", canonical_name="Rotational artificial gravity", aliases=["artificial gravity"],
                definition="Apparent gravity produced by rotation.", domain="physics",
                mechanism="Rotation produces centripetal acceleration that presses occupants toward the habitat's outer surface.",
                concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="This fabricated sentence is not present in the initial source corpus at all.")],
                curiosity_hook="Could rotation create gravity by moving a habitat's floor beneath its inhabitants?", uncertainty="",
            )], insufficient_evidence_reasons=[])

        def recover_candidate_evidence(self, candidate: EvidenceConceptCandidate, _synthesis: str, sources: list[dict[str, str]]) -> EvidenceRecoveryResult:
            return EvidenceRecoveryResult(candidate_id=candidate.candidate_id, recovered=True, reason="The recovery source defines the mechanism.", concept_evidence=[EvidenceReference(source_id=sources[0]["source_id"], excerpt="Rotational artificial gravity uses centripetal acceleration to press occupants toward the outer surface of a rotating habitat.")])

        def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
            candidate = candidates.candidates[0]
            return ConceptValidationBatch(results=[ValidatedConcept(
                candidate_id=candidate.candidate_id, accepted=True, rejection_reason="", discovery_origin=candidate.discovery_origin,
                discovery_source_ids=candidate.discovery_source_ids, research_path=candidate.research_path, label_kind=candidate.label_kind,
                canonical_name=candidate.canonical_name, aliases=candidate.aliases, level="mechanism", definition=candidate.definition,
                concept_evidence=candidate.concept_evidence, domain=candidate.domain, mechanism=candidate.mechanism,
                relationship_type="illustrated", relationship_evidence="The work depicts an engineered environment governed by rotational mechanics.",
                concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.9,
                relationship_source_ids=[work_sources[0]["source_id"]], curiosity_hook=candidate.curiosity_hook,
            )])

    research = RecoveryResearch()
    app.dependency_overrides[get_ai_client] = lambda: RecoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: research
    response = client.post(f"/api/items/{item_id}/enrich")
    assert response.status_code == 200
    assert research.calls == 1
    recovered = next(concept for concept in response.json()["concepts"] if concept["concept_classification"] == "established_concept")
    assert recovered["recovery_attempt"]["existing_corpus_review_attempted"] is True
    assert recovered["recovery_attempt"]["targeted_retrieval_attempted"] is True
    assert recovered["final_validation_outcome"] == "Accepted after final validation."
    assert "Targeted evidence recovery passed" in recovered["recovery_attempt"]["outcome"]
    assert "evidence_recovery" in {source["source_type"] for source in response.json()["sources"]}


def test_missing_recovery_evidence_remains_rejected_with_audit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCE_RECOVERY_MAX_CANDIDATES", "1")
    item_id = create_item(client).json()["id"]

    class MissingRecoveryResearch(FakeResearch):
        def recover_candidate(self, _item: dict[str, object], _candidate: dict[str, object]) -> ResearchPacket:
            source = RetrievedSource(url="https://example.edu/unrelated", title="Unrelated policy", excerpt="This source discusses governance policy and contains no account of rotating habitats or artificial gravity.")
            return ResearchPacket(synthesis="No relevant evidence found.", sources=[source])

    class MissingRecoveryAI(FakeAI):
        def extract_evidence_concepts(self, problems: ResearchProblemBatch, _synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
            return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(
                candidate_id="missing", problem_ids=[problems.problems[0].problem_id], discovery_origin="work_grounded",
                discovery_source_ids=[concept_sources[0]["source_id"]], research_path="Research into an engineered environment raised a concrete question about rotational artificial gravity.",
                label_kind="established_term", canonical_name="Rotational artificial gravity", aliases=["artificial gravity"], definition="Apparent gravity produced by rotation.", domain="physics",
                mechanism="Rotation produces centripetal acceleration that presses occupants toward the habitat's outer surface.",
                concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="A fabricated excerpt that does not occur in the initial source.")], curiosity_hook=None, uncertainty="",
            )], insufficient_evidence_reasons=[])

        def recover_candidate_evidence(self, candidate: EvidenceConceptCandidate, _synthesis: str, _sources: list[dict[str, str]]) -> EvidenceRecoveryResult:
            return EvidenceRecoveryResult(candidate_id=candidate.candidate_id, recovered=False, reason="The bounded search found no supporting definition.", concept_evidence=[])

        def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, _work: list[dict[str, str]], _sources: list[dict[str, str]]) -> ConceptValidationBatch:
            candidate = candidates.candidates[0]
            return ConceptValidationBatch(results=[ValidatedConcept(candidate_id=candidate.candidate_id, accepted=False, rejection_reason="Independent evidence remains missing after one recovery attempt.", discovery_origin=candidate.discovery_origin, discovery_source_ids=candidate.discovery_source_ids, research_path=candidate.research_path, label_kind=candidate.label_kind, canonical_name=candidate.canonical_name, aliases=candidate.aliases, level="mechanism", definition=candidate.definition, concept_evidence=candidate.concept_evidence, domain=candidate.domain, mechanism=candidate.mechanism, relationship_type="unsupported", relationship_evidence="", concept_verification="unverified", relationship_verification="unverified", confidence=0.2, relationship_source_ids=[], curiosity_hook=None)])

    app.dependency_overrides[get_ai_client] = lambda: MissingRecoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: MissingRecoveryResearch()
    response = client.post(f"/api/items/{item_id}/enrich")
    assert response.status_code == 200
    assert not [concept for concept in response.json()["concepts"] if concept["concept_classification"] == "established_concept"]
    assert any(concept["concept_classification"] == "grounded_observation" for concept in response.json()["concepts"])
    rejected = response.json()["rejected_leads"][0]
    assert rejected["recovery_attempt"]["targeted_retrieval_attempted"] is True
    assert "no supporting definition" in rejected["recovery_attempt"]["recovery_reason"]
    assert "remains missing" in rejected["final_validation_outcome"]


def test_canonical_dedupes_established_aliases_but_not_descriptive_aliases() -> None:
    test_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    test_session = sessionmaker(bind=test_engine)
    with test_session() as db:
        established = resolve_canonical_concept(db, "Bayesian inference", ["Bayesian reasoning"], "A definition.", "statistics", "Evidence updates prior probabilities over competing explanations.", "verified", "established_term")
        alias = resolve_canonical_concept(db, "Bayesian reasoning", [], "A definition.", "statistics", "Evidence updates prior probabilities over competing explanations.", "verified", "established_term")
        first_descriptive = resolve_canonical_concept(db, "Learning from failed predictions", ["Prediction-led learning"], "A definition.", "learning", "Prediction errors expose model weaknesses and guide revision.", "verified", "descriptive_label")
        second_descriptive = resolve_canonical_concept(db, "Prediction-led learning", ["Learning from failed predictions"], "A definition.", "learning", "Prediction errors expose model weaknesses and guide revision.", "verified", "descriptive_label")
        assert established.id == alias.id
        assert first_descriptive.id != second_descriptive.id
        assert len(db.scalars(select(CanonicalConcept)).all()) == 3


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
