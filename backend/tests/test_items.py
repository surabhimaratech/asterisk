from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import AIProviderError, CandidateBatch, ConceptExtraction, ConceptValidationBatch, CritiquedConnection, EnrichedConcept, EnrichmentAnalysis, EvidenceConceptBatch, EvidenceConceptCandidate, EvidenceRecoveryResult, EvidenceReference, ExtractedConcept, ConnectionCandidate, ResearchProblem, ResearchProblemBatch, ValidatedConcept, WorkUnderstanding
from app.main import BACKEND_DIR, DATA_DIR, DB_PATH, Base, CanonicalConcept, app, get_ai_client, get_db, resolve_canonical_concept
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

    def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]]) -> ResearchProblemBatch:
        return ResearchProblemBatch(problems=[ResearchProblem(problem_id="problem-1", concrete_description="A community changes its institutional rules.", work_situation="The work compares changing social arrangements.", intellectual_question="How do communities test alternative institutional forms?", relevant_domains=["political anthropology"], evidence_source_ids=[work_sources[0]["source_id"]], research_queries=["established term institutional forms testing collective rules"])])

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
    def understand_work(self, _item: dict[str, object], _synthesis: str, _sources: list[dict[str, str]]) -> WorkUnderstanding:
        return WorkUnderstanding(ambiguous_identity=True, confirmation_question="Which edition or creator did you mean?", resolved_title="", creator="", release_info="", identity_summary="", events_and_problems=[], phenomena_and_processes=[], dilemmas_and_decisions=[], systems_and_causal_relations=[])


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
    assert {concept["name"] for concept in second.json()["concepts"]} == {"My lens", "Institutional experimentation"}
    enriched = next(concept for concept in second.json()["concepts"] if concept["origin"] == "ai")
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


def test_research_problem_cannot_be_kept_as_a_concept(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: InsufficientEvidenceAI()
    finding = client.post(f"/api/items/{item_id}/enrich").json()["rejected_leads"][0]
    response = client.patch(f"/api/rejected-findings/{finding['id']}/review", json={"decision": "keep"})
    assert response.status_code == 422


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
    assert not [concept for concept in response.json()["concepts"] if concept["origin"] == "ai"]
    rejected = response.json()["rejected_leads"]
    assert rejected[0]["stage"] == "research_problem"
    assert "Insufficient evidence" in rejected[0]["rejection_reason"]
    assert "do not define an independent concept" in rejected[0]["rejection_reason"]


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


def test_verified_alias_reuses_canonical_without_string_similarity_merge(client: TestClient) -> None:
    first = create_item(client).json()["id"]
    second = create_item(client, title="A second work").json()["id"]
    app.dependency_overrides[get_research_provider] = lambda: FakeResearch()
    app.dependency_overrides[get_ai_client] = lambda: FakeAI()
    first_ai = next(concept for concept in client.post(f"/api/items/{first}/enrich").json()["concepts"] if concept["origin"] == "ai")

    app.dependency_overrides[get_ai_client] = lambda: AliasAI()
    second_ai = next(concept for concept in client.post(f"/api/items/{second}/enrich").json()["concepts"] if concept["origin"] == "ai")

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
    problem = ResearchProblem(problem_id="p1", concrete_description="Investigators infer a system's purpose from behavior.", work_situation="The crew observes an unfamiliar mechanism.", intellectual_question="How can function be inferred without specifications?", relevant_domains=["philosophy of science", "engineering"], evidence_source_ids=["work-1"], research_queries=["inferring system function from observed behavior without specifications"])
    assert research_problem_rejection(problem, {"work-1"}, "A Fictional Work") is None

    invented_source = problem.model_copy(update={"evidence_source_ids": ["work-altered"]})
    assert "not present" in (research_problem_rejection(invented_source, {"work-1"}, "A Fictional Work") or "")

    title_anchored = problem.model_copy(update={"research_queries": ["A Fictional Work abductive reasoning"]})
    assert "work title" in (research_problem_rejection(title_anchored, {"work-1"}, "A Fictional Work") or "")


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
        candidate_id="descriptive-1", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into observed behavior led to a source explaining how people infer possible goals from incomplete evidence.", label_kind="descriptive_label",
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
        candidate_id="descriptive-3", accepted=True, rejection_reason="", discovery_origin="work_grounded", discovery_source_ids=["concept-1"], research_path="Research into incomplete observations led to a source discussing provisional explanations under uncertainty.", label_kind="descriptive_label",
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
    recovered = next(concept for concept in response.json()["concepts"] if concept["origin"] == "ai")
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
    assert not [concept for concept in response.json()["concepts"] if concept["origin"] == "ai"]
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
