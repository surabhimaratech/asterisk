from collections.abc import Generator
from io import StringIO
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import AIProviderError, CandidateBatch, ConceptExtraction, ConceptValidationBatch, ConsideredDiscoveryFinding, CritiquedConnection, DiscoveryDirection, DiscoveryDirectionBatch, EnrichedConcept, EnrichmentAnalysis, EntityCandidate, EntityResolution, EvidenceConceptBatch, EvidenceConceptCandidate, EvidenceRecoveryResult, EvidenceReference, ExtractedConcept, ConnectionCandidate, ResearchProblem, ResearchProblemBatch, ResearchedDiscoveryClaim, ResearchedDiscoveryDraft, SidequestCandidateAssessment, SidequestClaim, ValidatedConcept, WorkUnderstanding, strict_json_schema
from app.discovery import NoDiscoveryError, run_researched_discovery
from app.main import BACKEND_DIR, DATA_DIR, DB_PATH, Base, CanonicalConcept, ItemConcept, SavedDiscovery, app, concepts_are_semantic_duplicates, configure_application_logging, deduplicate_concept_candidates, get_ai_client, get_db, migrate_additive_columns, resolve_canonical_concept, select_diverse_anchors
from app.prompts import (
    CONNECTION_SYSTEM_PROMPT,
    DISCOVERY_DIRECTION_SYSTEM_PROMPT,
    DISCOVERY_SYNTHESIS_SYSTEM_PROMPT,
    discovery_research_input,
)
from app.quality import assess_research_problem, candidate_evidence_rejection, concept_quality_rejection, deterministic_rejection_reason, rebind_exact_evidence, research_problem_rejection
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


def test_saved_discoveries_survive_schema_upgrade_and_database_reopen(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'existing.db'}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine, tables=[
        table for table in Base.metadata.sorted_tables if table is not SavedDiscovery.__table__
    ])
    testing_session = sessionmaker()

    def override_get_db():
        with testing_session() as session:
            yield session

    def provider_must_not_be_called():
        pytest.fail("Saved discovery routes must not use research providers")

    monkeypatch.setitem(app.dependency_overrides, get_db, override_get_db)
    monkeypatch.setitem(app.dependency_overrides, get_ai_client, provider_must_not_be_called)
    monkeypatch.setitem(app.dependency_overrides, get_research_provider, provider_must_not_be_called)
    testing_session.configure(bind=engine)
    with TestClient(app) as client:
        item = create_item(client).json()
        Base.metadata.create_all(engine)
        assert client.get("/api/items").json() == [item]
        path = "/api/discoveries/saved"
        snapshot = {
            "candidate_id": "flood-mark", "hook": "The flood mark",
            "story": "A building retained a historical flood mark.",
            "go_deeper": "Who preserved the mark?",
            "evidence": [{"source_id": "archive", "url": "https://example.org", "title": "Archive", "excerpt": "The mark was retained."}],
        }
        invalid = {**snapshot, "evidence": [{**snapshot["evidence"][0], "url": "javascript:alert(1)"}]}
        assert client.post(path, json=[snapshot, invalid]).status_code == 422
        assert client.get(path).json() == []

        response = client.post(path, json=[snapshot, snapshot])
        assert response.status_code == 200
        saved = response.json()
        assert len(saved) == 1
        assert saved[0] == {**snapshot, "saved_at": saved[0]["saved_at"]}
        assert client.post(path, json=[{**snapshot, "hook": "Duplicate"}]).json() == saved
        newer = {**snapshot, "candidate_id": "newer"}
        assert [entry["candidate_id"] for entry in client.post(path, json=[newer]).json()] == ["newer", "flood-mark"]
        assert client.delete(f"/api/items/{item['id']}").status_code == 204

    engine.dispose()
    reopened = create_engine(database_url)
    testing_session.configure(bind=reopened)
    try:
        with TestClient(app) as fresh_client:
            assert fresh_client.get(path).json()[1:] == saved
            assert fresh_client.delete(f"{path}/flood-mark").status_code == 204
            assert fresh_client.delete(f"{path}/flood-mark").status_code == 204
            assert [entry["candidate_id"] for entry in fresh_client.get(path).json()] == ["newer"]
    finally:
        reopened.dispose()


def considered_discovery_finding(
    finding: str = "A concrete documented finding.", source_ids: list[str] | None = None,
    *, selected: bool = True, direction_index: int = 0, story: str | None = None,
    twist: str = "The documented outcome complicates the expectation established by the case.",
    library_connection: str | None = None,
) -> ConsideredDiscoveryFinding:
    return ConsideredDiscoveryFinding(
        direction_index=direction_index, hook=finding,
        story=story or f"{finding} The retrieved record documents the particular event and its outcome.",
        twist=twist, library_connection=library_connection,
        evidence_assessment="The cited excerpt directly supports the stated particular.",
        go_deeper="Which primary record would reveal what happened next?",
        source_ids=["discovery-source-1"] if source_ids is None else source_ids, selected=selected,
    )


def test_default_database_path_is_stable() -> None:
    assert DB_PATH == BACKEND_DIR / "data" / "asterisk.db"
    assert DB_PATH.is_absolute()
    assert DATA_DIR.is_dir()


def test_application_diagnostics_use_the_server_terminal_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASTERISK_LOG_LEVEL", raising=False)
    output = StringIO()
    handler = logging.StreamHandler(output)
    server_parent = logging.Logger("test.uvicorn")
    server_parent.addHandler(handler)
    server_parent.propagate = False
    server_logger = logging.Logger("test.uvicorn.error")
    server_logger.parent = server_parent
    server_logger.propagate = True
    target_logger = logging.Logger("test.asterisk")

    configure_application_logging(target_logger, server_logger)
    target_logger.info('sidequest_diagnostic {"stage": "generation"}')

    assert target_logger.level == logging.INFO
    assert target_logger.propagate is False
    assert 'sidequest_diagnostic {"stage": "generation"}' in output.getvalue()


def test_work_understanding_strict_schema_is_closed_without_profile_dictionary() -> None:
    schema = strict_json_schema(WorkUnderstanding.model_json_schema())

    assert "curiosity_profile" not in schema["properties"]

    def assert_declared_objects_are_closed(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                assert_declared_objects_are_closed(value)
        elif isinstance(node, list):
            for value in node:
                assert_declared_objects_are_closed(value)

    assert_declared_objects_are_closed(schema)


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

    def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
        context = contexts[0]
        clue_item = context["item_a"]
        a = clue_item["id"]  # type: ignore[index]
        a_concept = clue_item["concepts"][0]["name"]  # type: ignore[index]
        first = ConnectionCandidate(
            item_a_id=a, item_b_id=a, title="Coordination without a conductor",
            connection="Local agents acting on incomplete signals can produce organized collective behavior without global oversight.",
            bridge="Decentralized coordination under incomplete information", why_interesting="It reframes control as an emergent property.",
            where_it_breaks="The biological system adapts through evolution while the engineered system follows designed rules.",
            new_question="When does local coordination become resilient rather than chaotic?",
            origin_item_ids=[a], clue=f"{a_concept} begins with local actors responding to incomplete signals.",
            clue_item_id=a, clue_concept_names=[a_concept], lead="Distributed cognition under partial information.",
            lead_kind="unfamiliar_subject", lead_category="intellectual_problem", reasoning_steps=[
                "The first clue replaces central control with local responses to partial information.",
                "Distributed cognition asks how those partial views can still produce coordinated behavior.",
            ], claims=[
                SidequestClaim(claim_type="interpretation", text="The shared mechanism may be coordination without global oversight."),
                SidequestClaim(claim_type="speculative_hypothesis", text="The comparison may reveal when decentralization becomes resilient."),
            ],
        )
        second = ConnectionCandidate(
            item_a_id=a, item_b_id=a, title="The threshold where a group commits",
            connection="The clue can open a question about how local signals accumulate until a collective system changes state.",
            bridge="Threshold aggregation converting local signals into collective commitment",
            why_interesting="It shifts attention from coordination itself to the moment a scattered pattern becomes a decision.",
            where_it_breaks="A formal threshold model may not capture evolving goals, unequal influence, or ambiguous signals.",
            new_question="Which quorum-like systems change behavior only after enough independent local signals agree?",
            origin_item_ids=[a], clue=f"{a_concept} begins with local actors responding to incomplete signals.",
            clue_item_id=a, clue_concept_names=[a_concept], lead="Quorum thresholds in collective decision-making.",
            lead_kind="unfamiliar_subject", lead_category="phenomenon", reasoning_steps=[
                "Separate actors contribute signals without seeing the whole system.",
                "A threshold can convert enough local agreement into a collective change of state.",
            ], claims=[
                SidequestClaim(claim_type="interpretation", text="The clue can be read as a threshold problem rather than only a coordination problem."),
                SidequestClaim(claim_type="speculative_hypothesis", text="Quorum-like thresholds may offer a useful external direction."),
            ],
        )
        return CandidateBatch(candidates=[first, second])

    def critique_candidates(self, candidates: list[dict[str, object]], _dossiers: list[dict[str, object]]) -> CritiquedConnection:
        candidate = candidates[0]
        return CritiquedConnection(
            accepted=not self.reject, rejection_reason="Too superficial" if self.reject else "",
            **candidate, evidence_status="Exploratory and not externally verified.", concepts_used=candidate["clue_concept_names"], evidence=[],
            selected_candidate_index=0 if not self.reject else None,
            candidate_assessments=[SidequestCandidateAssessment(
                candidate_index=index, external_lead_aligned=True, mechanism_explained=True,
                next_investigation_follows_lead=True, valid=True, rejection_reason="",
            ) for index, _candidate in enumerate(candidates)],
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

    def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]], _curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
        return ResearchProblemBatch(problems=[ResearchProblem(problem_id="problem-1", concrete_description="A community changes its institutional rules.", work_situation="The work compares changing social arrangements.", intellectual_question="How do communities test alternative institutional forms?", relevant_domains=["political anthropology"], evidence_source_ids=[work_sources[0]["source_id"]], research_queries=["established term institutional forms testing collective rules"], anchor_kind="system", concise_label="Testing social arrangements", interesting_reason="The work treats political organization as something communities can deliberately vary rather than a fixed evolutionary stage.", distinctiveness="The comparison centers concrete institutional experiments instead of a generic theme of social change.", discovery_mode="deepen_fascination", external_research_warranted=True, discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "medium", "rationale": "A concrete institutional mechanism with cross-domain potential."}, mechanism="Changing social arrangements changes institutional rules, incentives, and collective behavior.", curiosity_invitation="When does changing an institutional rule reveal who benefited from the previous social arrangement?")])

    def extract_evidence_concepts(self, problems: ResearchProblemBatch, _concept_synthesis: str, concept_sources: list[dict[str, str]]) -> EvidenceConceptBatch:
        return EvidenceConceptBatch(candidates=[EvidenceConceptCandidate(candidate_id="candidate-1", problem_ids=[problems.problems[0].problem_id], discovery_origin="work_grounded", discovery_source_ids=[concept_sources[0]["source_id"]], research_path="Research into how communities test alternate rules led to an independent account of institutional experimentation.", label_kind="established_term", canonical_name="Institutional experimentation", aliases=["institutional experiments"], definition="Deliberate or emergent testing of institutional forms.", domain="political anthropology", mechanism="Rules change incentives and collective behavior.", concept_evidence=[EvidenceReference(source_id=concept_sources[0]["source_id"], excerpt="Institutional experimentation tests alternate rules and arrangements in collective life.")], curiosity_hook="When do changes in collective rules become durable institutional experiments?", uncertainty="The term is used differently across disciplines.")], insufficient_evidence_reasons=[])

    def validate_concepts(self, _understanding: WorkUnderstanding, _problems: ResearchProblemBatch, candidates: EvidenceConceptBatch, _work_sources: list[dict[str, str]], _concept_sources: list[dict[str, str]]) -> ConceptValidationBatch:
        candidate = candidates.candidates[0]
        return ConceptValidationBatch(results=[ValidatedConcept(candidate_id=candidate.candidate_id, accepted=True, rejection_reason="", discovery_origin=candidate.discovery_origin, discovery_source_ids=candidate.discovery_source_ids, research_path=candidate.research_path, label_kind=candidate.label_kind, canonical_name=candidate.canonical_name, aliases=candidate.aliases, level="concept", definition=candidate.definition, concept_evidence=candidate.concept_evidence, domain=candidate.domain, mechanism=candidate.mechanism, relationship_type="illustrated", relationship_evidence="The work compares changes in social organization.", concept_verification="verified", relationship_verification="supported_interpretation", confidence=0.84, relationship_source_ids=[_work_sources[0]["source_id"]], curiosity_hook=candidate.curiosity_hook)])


class FakeResearch:
    def research(self, _item: dict[str, object]) -> ResearchPacket:
        source = RetrievedSource(
            url="https://example.com/publisher", title="Publisher description",
            excerpt="A community changes its institutional rules. The work compares changing social arrangements and shows how rules alter incentives and collective behavior.",
        )
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
    assert created.json()["evidence_status"].startswith("Exploratory")
    assert client.get("/api/sidequests").json() == [created.json()]


def test_researched_discovery_is_cited_standalone_and_not_persisted(client: TestClient) -> None:
    item_id = create_item(client, title="Piranesi", note="The journals and tides stayed with me.").json()["id"]
    client.post(f"/api/items/{item_id}/concepts", json={"name": "External memory", "category": "mechanism"})

    class DiscoveryAI(FakeAI):
        seed: dict[str, object] = {}
        directions: list[dict[str, object]] = []

        def propose_discovery_directions(self, seed: dict[str, object]) -> DiscoveryDirectionBatch:
            self.seed = seed
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="How have tide tables served as civic memory?", seed_detail="Journals record recurring tides.",
                    why_promising="It turns private notation into shared prediction.",
                    discovery_target="A documented port decision or outcome caused by an early published tide table.",
                    search_queries=["history tide tables civic navigation"],
                ),
                DiscoveryDirection(
                    question="Which buildings encode recurring water levels?", seed_detail="The setting is organized by tidal recurrence.",
                    why_promising="Architecture can make environmental memory visible.",
                    discovery_target="A specific building feature that records a historic flood level and changed later decisions.",
                    search_queries=["architecture marks historic flood levels"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            self.directions = directions
            assert sources[0]["source_id"] == "discovery-source-1"
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="The tide table that became public infrastructure",
                discovery="Nineteenth-century tide prediction moved recurring observations into published tables used by ports.",
                why_surprising="A repetitive notebook practice became infrastructure for coordinating strangers.",
                connection_to_library=None,
                what_remains_uncertain="The retrieved source does not establish how uniformly ports adopted the tables.",
                next_rabbit_hole="Which port first treated tide prediction as a public service?",
                claims=[
                    ResearchedDiscoveryClaim(
                        claim_type="supported_fact", text="Published tide tables supported port navigation.",
                        source_ids=["discovery-source-1"],
                    ),
                    ResearchedDiscoveryClaim(
                        claim_type="interpretation", text="A record can become coordination infrastructure.", source_ids=[],
                    ),
                ],
                considered_findings=[considered_discovery_finding("Published tide tables became port infrastructure.")],
                selection_rationale="This was the most concrete source-supported institutional outcome.",
            )

    class DiscoveryResearch:
        captured_directions: list[dict[str, object]] = []

        def research_discovery(self, seed, directions) -> ResearchPacket:
            self.captured_directions = directions
            return ResearchPacket(synthesis="An institutional history describes published tide tables.", sources=[RetrievedSource(
                url="https://example.org/archive/tide-tables", title="A history of tide prediction",
                excerpt="Published tide tables became an aid to navigation and port planning.",
            )])

    ai = DiscoveryAI()
    research = DiscoveryResearch()
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: research

    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    body = response.json()
    assert body["connection_to_library"] is None
    assert body["persisted"] is False
    assert body["evidence"] == [{
        "source_id": "discovery-source-1", "url": "https://example.org/archive/tide-tables",
        "title": "A history of tide prediction",
        "excerpt": "Published tide tables became an aid to navigation and port planning.",
    }]
    assert body["claims"][0]["source_ids"] == ["discovery-source-1"]
    assert ai.seed["note"] == "The journals and tides stayed with me."
    assert len(research.captured_directions) == 2
    assert client.get("/api/sidequests").json() == []


def test_researched_discovery_returns_no_result_without_credible_sources(client: TestClient) -> None:
    item_id = create_item(client, kind="thought", title="Maps remember hesitation").json()["id"]

    class DirectionAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="Where are pauses mapped?", why_promising="A traceable practice.", seed_detail="Hesitation.", discovery_target="A map or study that records a pause as data.", search_queries=["mapped pauses"]),
                DiscoveryDirection(question="How do maps record uncertainty?", why_promising="A documented technique.", seed_detail="Uncertain routes.", discovery_target="A specific cartographic mark invented to show uncertainty.", search_queries=["cartography uncertainty"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources):
            raise AssertionError("Synthesis must not run without inspectable sources.")

    class EmptyResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Nothing credible was retrieved.", sources=[])

    app.dependency_overrides[get_ai_client] = lambda: DirectionAI()
    app.dependency_overrides[get_research_provider] = lambda: EmptyResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 422
    assert "No researched discovery yet" in response.json()["detail"]
    assert client.get("/api/sidequests").json() == []


def test_researched_discovery_reports_search_failure(client: TestClient) -> None:
    item_id = create_item(client, kind="concept", title="Distributed systems").json()["id"]

    class DirectionAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="How do crews coordinate?", why_promising="Documented cases exist.", seed_detail="Coordination.", discovery_target="A specific coordination practice and documented outcome.", search_queries=["crew coordination case"]),
                DiscoveryDirection(question="Where do local rules fail?", why_promising="Failure reports are searchable.", seed_detail="Local rules.", discovery_target="A failure report identifying a counterintuitive local-rule breakdown.", search_queries=["local rules failure report"]),
            ])

    class FailingResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            raise AIProviderError("Discovery research timed out. Existing data was preserved.")

    app.dependency_overrides[get_ai_client] = lambda: DirectionAI()
    app.dependency_overrides[get_research_provider] = lambda: FailingResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 502
    assert "timed out" in response.json()["detail"]


def test_researched_discovery_rejects_invented_source_attribution(client: TestClient) -> None:
    item_id = create_item(client, kind="article", title="An unusual archive").json()["id"]

    class InventedCitationAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="What did the archive preserve?", why_promising="A documented collection.", seed_detail="The archive.", discovery_target="One unusual object preserved by the archive and why it survived.", search_queries=["unusual archive collection"]),
                DiscoveryDirection(question="Who used the collection?", why_promising="Institutional records may answer it.", seed_detail="Its users.", discovery_target="A documented instance of an unexpected person using the collection.", search_queries=["archive collection users"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="An unsupported discovery",
                discovery="The result cites a source the search never returned.", why_surprising="It sounds convincing.",
                connection_to_library=None, what_remains_uncertain="The citation is missing.",
                next_rabbit_hole="Where is the real record?", claims=[ResearchedDiscoveryClaim(
                    claim_type="supported_fact", text="An unsupported factual claim.",
                    source_ids=["discovery-source-99"],
                )],
                considered_findings=[considered_discovery_finding(
                    "An unsupported finding.", ["discovery-source-99"],
                )],
                selection_rationale="The invented citation appeared to support it.",
            )

    class OneSourceResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="One source was retrieved.", sources=[RetrievedSource(
                url="https://example.org/real-source", title="Real source", excerpt="A real but different record.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: InventedCitationAI()
    app.dependency_overrides[get_research_provider] = lambda: OneSourceResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 422
    assert "survived source validation" in response.json()["detail"]
    assert client.get("/api/sidequests").json() == []


def test_researched_discovery_seed_excludes_concepts_beyond_spoiler_ceiling(client: TestClient) -> None:
    item_id = create_item(client, title="Unfinished novel", progress_status="planned", spoiler_policy="avoid").json()["id"]
    client.post(f"/api/items/{item_id}/concepts", json={"name": "Premise-safe clue", "category": "general"})
    override = app.dependency_overrides[get_db]
    session_generator = override()
    session = next(session_generator)
    session.add(ItemConcept(
        item_id=item_id, name="Ending revelation", category="general", origin="ai", spoiler_scope="full",
    ))
    session.commit()
    session.close()

    class CapturingAI(FakeAI):
        seed: dict[str, object] = {}

        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            self.seed = seed
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="What practices echo the premise?", why_promising="A safe lead.", seed_detail="Premise.", discovery_target="A documented practice with one observable consequence.", search_queries=["premise practice"]),
                DiscoveryDirection(question="Which archives preserve similar rules?", why_promising="An archival lead.", seed_detail="Premise rule.", discovery_target="A specific archival record of the rule being used.", search_queries=["archive premise rule"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="A safe discovery", discovery="An archive documents the practice.",
                why_surprising="The practice survived in an unexpected setting.", connection_to_library="The premise supplies the question.",
                what_remains_uncertain="The source covers one archive.", next_rabbit_hole="Which second archive records it?",
                claims=[ResearchedDiscoveryClaim(claim_type="supported_fact", text="The archive documents it.", source_ids=["discovery-source-1"])],
                considered_findings=[considered_discovery_finding("The archive documents the practice.")],
                selection_rationale="The archive supplied the strongest concrete record.",
            )

    class SafeResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Archive evidence.", sources=[RetrievedSource(
                url="https://example.org/archive", title="Archive", excerpt="The archive documents the practice.",
            )])

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: SafeResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    assert ai.seed["spoiler_ceiling"] == "premise_only"
    assert [concept["name"] for concept in ai.seed["concepts"]] == ["Premise-safe clue"]


def test_discovery_quality_contract_seeks_particulars_without_keyword_rules() -> None:
    direction = DiscoveryDirection(
        question="What failed when the signal was repeated?", why_promising="The exception may expose a mechanism.",
        seed_detail="A repeated signal.", discovery_target="A documented experiment with an unexpected failure mode.",
        search_queries=["repeated signal unexpected experiment failure"],
    )
    research_input = discovery_research_input({"title": "A seed"}, [direction.model_dump()])
    direction_contract = " ".join(DISCOVERY_DIRECTION_SYSTEM_PROMPT.split())
    synthesis_contract = " ".join(DISCOVERY_SYNTHESIS_SYSTEM_PROMPT.split())

    assert "curiosity hooks" in direction_contract
    assert "identifiable artifact, record, incident" in direction_contract
    assert "minimum information a retrieved excerpt would need to contain" in direction_contract
    assert "focused, case-finding search query" in direction_contract
    assert "search for a category of documented cases" in direction_contract
    assert "never invent a name" in direction_contract
    assert "Adding words such as “interesting” or “surprising”" in direction_contract
    assert "particular incidents" in research_input
    assert "contradict a direction's premise" in research_input
    assert "Being unfamiliar to the user is not by itself enough" in synthesis_contract
    assert "return accepted false" in synthesis_contract
    assert "considered_findings" in synthesis_contract
    assert "specificity" in synthesis_contract
    assert "counterintuitiveness" in synthesis_contract
    assert "miniature, evidence-backed stories" in synthesis_contract
    assert "More than one may emerge from the same research direction" in synthesis_contract
    assert "intriguing title wrapped around an ordinary claim does not qualify" in synthesis_contract
    assert "Never invent an incident, quotation" in synthesis_contract
    assert "information actually stated in the retrieved excerpts" in synthesis_contract
    assert "citation metadata" in synthesis_contract
    assert "broad topical relevance is not evidence" in synthesis_contract
    assert "Source IDs establish provenance" in synthesis_contract
    assert "their presence alone does not establish" in synthesis_contract
    assert "Do not favor a candidate because it is closer to the seed's genre" in synthesis_contract


def test_researched_discovery_declines_generic_sourced_generalization(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class GenericFindingAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="Did humor ever produce an unexpected outcome in an isolated crew?",
                    why_promising="A specific incident could reveal more than a broad coping claim.",
                    seed_detail="Humor during crisis.",
                    discovery_target="A documented incident or study result where humor changed group behavior unexpectedly.",
                    search_queries=["isolated crew humor unexpected outcome documented"],
                ),
                DiscoveryDirection(
                    question="When did joking make a crisis worse rather than better?",
                    why_promising="A counterexample could expose the limits of the familiar claim.",
                    seed_detail="Humor during crisis.",
                    discovery_target="A documented counterexample in which humor disrupted trust or coordination.",
                    search_queries=["crisis humor disrupted trust documented case"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=False,
                rejection_reason="The excerpts support only the familiar generalization that humor can aid coping; they establish no distinctive incident, result, or mechanism.",
                title="", discovery="", why_surprising="", connection_to_library=None,
                what_remains_uncertain="", next_rabbit_hole="", claims=[],
                considered_findings=[considered_discovery_finding(
                    "Humor can support coping.", selected=False,
                )],
                selection_rationale="The only finding was familiar and insufficiently distinctive.",
            )

    class GenericResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="A review says humor may support coping and social bonding.", sources=[RetrievedSource(
                url="https://example.org/humor-review", title="Humor and coping review",
                excerpt="Humor can function as a coping strategy and support social bonding during stress.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: GenericFindingAI()
    app.dependency_overrides[get_research_provider] = lambda: GenericResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 422
    assert "familiar generalization" in response.json()["detail"]
    assert client.get("/api/sidequests").json() == []


def test_researched_discovery_accepts_concrete_finding_unrelated_to_seed_subject(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class OutwardFindingAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="Which color-making recipes depend on a living organism?",
                    why_promising="A material process may hide an unusual collaboration.", seed_detail="Working with unfamiliar life.",
                    discovery_target="A documented dye recipe whose color depends on a specific organism and preparation sequence.",
                    search_queries=["museum lichen dye historical recipe preparation"],
                ),
                DiscoveryDirection(
                    question="Which craft materials change only after a long rest?",
                    why_promising="Waiting may be an active part of manufacture.", seed_detail="Delayed problem solving.",
                    discovery_target="A documented craft process in which a resting interval causes a surprising material change.",
                    search_queries=["traditional craft resting period material transformation"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="A purple dye that develops in the dark",
                discovery="A museum record describes a lichen dye preparation sealed away for weeks before it develops its color.",
                why_surprising="The waiting period is part of making the color, not merely storage after the work is done.",
                connection_to_library=None,
                what_remains_uncertain="The excerpt does not establish how consistently historical dyers used this exact preparation.",
                next_rabbit_hole="Compare the museum's recipe with surviving dye notebooks from a named textile archive.",
                claims=[ResearchedDiscoveryClaim(
                    claim_type="supported_fact",
                    text="The documented preparation seals lichen mixture away for weeks before the purple develops.",
                    source_ids=["discovery-source-1"],
                )],
                considered_findings=[considered_discovery_finding(
                    "A lichen dye develops only after a sealed resting period.",
                )],
                selection_rationale="The specific material transformation was directly documented and counterintuitive.",
            )

    class DyeResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="A museum recipe records a delayed lichen-dye process.", sources=[RetrievedSource(
                url="https://example.org/museum/lichen-dye", title="Museum lichen dye record",
                excerpt="The preparation was sealed and left for several weeks before the purple color developed.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: OutwardFindingAI()
    app.dependency_overrides[get_research_provider] = lambda: DyeResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    assert response.json()["title"] == "A purple dye that develops in the dark"
    assert response.json()["connection_to_library"] is None
    assert response.json()["evidence"][0]["source_id"] == "discovery-source-1"


def test_researched_discovery_does_not_invent_a_particular_when_evidence_is_insufficient(client: TestClient) -> None:
    item_id = create_item(client, kind="thought", title="A clock that remembers weather").json()["id"]

    class InsufficientEvidenceAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="Did any public clock record weather rather than time?",
                    why_promising="A surviving instrument would be a concrete contradiction.", seed_detail="A weather-memory clock.",
                    discovery_target="A named clock or instrument that visibly accumulated a weather record.",
                    search_queries=["museum clock instrument records weather history"],
                ),
                DiscoveryDirection(
                    question="Which instruments preserve yesterday's atmosphere mechanically?",
                    why_promising="A mechanism may turn memory into a physical trace.", seed_detail="Remembering weather.",
                    discovery_target="A documented mechanical instrument and the physical trace it preserves.",
                    search_queries=["mechanical instrument preserves weather trace museum"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            assert sources[0]["excerpt"] == "The collection includes instruments used to observe weather."
            return ResearchedDiscoveryDraft(
                accepted=False,
                rejection_reason="The source mentions weather instruments but does not establish a named object, recording mechanism, or surprising outcome.",
                title="", discovery="", why_surprising="", connection_to_library=None,
                what_remains_uncertain="", next_rabbit_hole="", claims=[],
                considered_findings=[considered_discovery_finding(
                    "A museum collection contains weather instruments.", selected=False,
                )],
                selection_rationale="The excerpt did not establish a specific object, mechanism, or outcome.",
            )

    class ThinResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Only collection-level background was found.", sources=[RetrievedSource(
                url="https://example.org/museum/weather", title="Weather instruments collection",
                excerpt="The collection includes instruments used to observe weather.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: InsufficientEvidenceAI()
    app.dependency_overrides[get_research_provider] = lambda: ThinResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 422
    assert "does not establish a named object" in response.json()["detail"]


def test_synthesis_can_select_lenape_case_over_generic_humor_regardless_of_source_order(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class LiveTraceAI(FakeAI):
        direction_calls = 0
        synthesis_calls = 0

        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            self.direction_calls += 1
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="What has humor changed during a space mission?", why_promising="Mission records may contain an incident.",
                    seed_detail="Humor under pressure.", discovery_target="A documented mission incident with an observable outcome.",
                    search_queries=["space mission humor documented incident outcome"],
                ),
                DiscoveryDirection(
                    question="When has selective breeding made a crop more dangerous?", why_promising="A reversal may be documented.",
                    seed_detail="Selective breeding under crisis.", discovery_target="A named crop bred for one trait and withdrawn after a dangerous consequence.",
                    search_queries=["selective breeding crop withdrawn dangerous compound case"],
                ),
                DiscoveryDirection(
                    question="Which cooperation rules fail between species?", why_promising="A concrete case may expose a hidden tradeoff.",
                    seed_detail="Interspecies cooperation.", discovery_target="A documented interspecies practice with a counterintuitive ethical consequence.",
                    search_queries=["interspecies cooperation documented ethical conflict case"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            self.synthesis_calls += 1
            assert len(sources) == 6
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="The potato bred into a safety warning",
                discovery="The Lenape potato was commercially released after selective breeding, then withdrawn after tests found dangerously elevated solanine.",
                why_surprising="Breeding for a useful commercial crop produced a named variety whose natural defense chemistry made it unsafe.",
                connection_to_library=None,
                what_remains_uncertain="The excerpt does not isolate which selected trait caused the elevated solanine.",
                next_rabbit_hole="Trace the Lenape breeding records to learn which parent lines carried the high-solanine risk.",
                claims=[ResearchedDiscoveryClaim(
                    claim_type="supported_fact",
                    text="The Lenape potato was withdrawn after dangerously elevated solanine was detected.",
                    source_ids=["discovery-source-6"],
                )],
                considered_findings=[
                    ConsideredDiscoveryFinding(
                        direction_index=0, hook="Humor as a pressure valve",
                        story="Astronauts report using humor to cope with stress. The excerpt offers a broad pattern, not a particular incident or outcome.",
                        twist="The claim is familiar and does not complicate an expectation.", library_connection=None,
                        evidence_assessment="The astronaut source supports the broad pattern.",
                        go_deeper="Did a joke ever measurably alter a mission decision?",
                        source_ids=["discovery-source-1"], selected=False,
                    ),
                    ConsideredDiscoveryFinding(
                        direction_index=1, hook="The potato bred into a safety warning",
                        story="The Lenape potato was commercially released after selective breeding. It was later withdrawn when tests found dangerously elevated solanine.",
                        twist="Selection intended to improve a crop produced a dangerous commercial outcome.", library_connection=None,
                        evidence_assessment="The NIH excerpt directly reports the release, elevated solanine, and withdrawal.",
                        go_deeper="Which parent line introduced the chemical risk?",
                        source_ids=["discovery-source-6"], selected=True,
                    ),
                    ConsideredDiscoveryFinding(
                        direction_index=1, hook="When uniformity made a crop fragile",
                        story="Genetic uniformity increased potato-crop vulnerability to blight. A consistent crop population became a shared point of failure.",
                        twist="Agricultural consistency can amplify systemic fragility.", library_connection=None,
                        evidence_assessment="The Berkeley excerpt supports the vulnerability mechanism.",
                        go_deeper="Which modern crops retain the same uniformity risk?",
                        source_ids=["discovery-source-2"], selected=False,
                    ),
                ],
                selection_rationale="The Lenape case is the most concrete counterintuitive outcome and has direct evidence; humor is supported but generic.",
            )

    class LiveTraceResearch:
        calls = 0

        def research_discovery(self, seed, directions) -> ResearchPacket:
            self.calls += 1
            return ResearchPacket(synthesis="Six results covered all three directions.", sources=[
                RetrievedSource(url="https://example.org/space-humor", title="Humor in spaceflight", excerpt="Astronauts reported humor as one way of coping with stress."),
                RetrievedSource(url="https://example.org/berkeley-blight", title="Uniform crops and blight", excerpt="Genetic uniformity left potato crops vulnerable to widespread blight."),
                RetrievedSource(url="https://example.org/cooperation-one", title="Cooperation case one", excerpt="One documented practice involved cooperation across species."),
                RetrievedSource(url="https://example.org/cooperation-two", title="Cooperation case two", excerpt="A second source discussed an ethical tension."),
                RetrievedSource(url="https://example.org/space-history", title="Space mission history", excerpt="Mission crews developed informal social practices."),
                RetrievedSource(url="https://example.org/nih-lenape", title="The Lenape potato case", excerpt="Lenape was commercially released and later withdrawn after dangerously elevated solanine was found."),
            ])

    ai = LiveTraceAI()
    research = LiveTraceResearch()
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: research
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    assert response.json()["title"] == "The potato bred into a safety warning"
    assert response.json()["connection_to_library"] is None
    assert [source["source_id"] for source in response.json()["evidence"]] == ["discovery-source-6"]
    candidates = response.json()["discovery_candidates"]
    assert len(candidates) == 3
    selected = next(candidate for candidate in candidates if candidate["ai_suggested"])
    assert selected["hook"] == "The potato bred into a safety warning"
    assert "commercially released" in selected["story"]
    assert "dangerous commercial outcome" in selected["twist"]
    assert selected["library_connection"] is None
    assert selected["evidence"][0]["source_id"] == "discovery-source-6"
    crop_candidates = [candidate for candidate in candidates if candidate["research_direction"] == "When has selective breeding made a crop more dangerous?"]
    assert len(crop_candidates) == 2  # distinct little stories may come from one productive direction
    assert response.json()["suggested_candidate_id"] == selected["candidate_id"]
    assert ai.direction_calls == ai.synthesis_calls == research.calls == 1
    assert client.get("/api/sidequests").json() == []


def test_synthesis_rejects_selected_finding_without_direct_source_support(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class UnsupportedWinnerAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="Which result reversed expectations?", why_promising="A reversal may be documented.", seed_detail="A reversal.", discovery_target="A named unexpected result.", search_queries=["named unexpected result"]),
                DiscoveryDirection(question="Which object was withdrawn?", why_promising="A withdrawal leaves records.", seed_detail="A safety failure.", discovery_target="A named object withdrawn for a documented reason.", search_queries=["product withdrawn documented reason"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="An unsupported specific story",
                discovery="A named object was supposedly withdrawn after an unusual reaction.",
                why_surprising="The reaction would be unexpected if established.", connection_to_library=None,
                what_remains_uncertain="The source does not establish the story.", next_rabbit_hole="Find a primary withdrawal record.",
                claims=[ResearchedDiscoveryClaim(
                    claim_type="supported_fact", text="The retrieved overview discusses product safety generally.",
                    source_ids=["discovery-source-1"],
                )],
                considered_findings=[considered_discovery_finding(
                    "A named object was withdrawn after an unusual reaction.", source_ids=[], selected=True,
                )],
                selection_rationale="The story sounded specific, but it lacked direct retrieved evidence.",
            )

    class ThinWinnerResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Only a general safety overview was retrieved.", sources=[RetrievedSource(
                url="https://example.org/safety", title="Safety overview",
                excerpt="Product safety processes can include withdrawal after hazards are identified.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: UnsupportedWinnerAI()
    app.dependency_overrides[get_research_provider] = lambda: ThinWinnerResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 422
    assert "survived source validation" in response.json()["detail"]


def test_discovery_candidates_exclude_invalid_selected_finding_but_return_supported_alternative(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class InvalidSelectionAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="Which crop failed unexpectedly?", why_promising="A named case may survive.", seed_detail="Selective breeding.", discovery_target="A named crop and documented outcome.", search_queries=["crop breeding unexpected outcome"]),
                DiscoveryDirection(question="Which mission ritual mattered?", why_promising="Logs may preserve one.", seed_detail="Crew culture.", discovery_target="A named mission ritual with an observed consequence.", search_queries=["mission ritual documented consequence"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="Unsupported winner", discovery="An unsupported mission story.",
                why_surprising="It would be surprising if sourced.", connection_to_library=None,
                what_remains_uncertain="No source establishes it.", next_rabbit_hole="Find a mission log.",
                claims=[ResearchedDiscoveryClaim(claim_type="supported_fact", text="Unsupported story.", source_ids=["discovery-source-99"])],
                considered_findings=[
                    considered_discovery_finding("An unsupported mission story.", ["discovery-source-99"], direction_index=1),
                    considered_discovery_finding(
                        "Genetic uniformity left potato crops vulnerable to blight.", ["discovery-source-1"],
                        selected=False, direction_index=0,
                    ),
                ],
                selection_rationale="The selected story was not actually present in the retrieved evidence.",
            )

    class AlternativeResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="One supported crop finding.", sources=[RetrievedSource(
                url="https://example.org/blight", title="Crop uniformity and blight",
                excerpt="Genetic uniformity left the potato crop vulnerable to widespread blight.",
            )])

    app.dependency_overrides[get_ai_client] = lambda: InvalidSelectionAI()
    app.dependency_overrides[get_research_provider] = lambda: AlternativeResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] is None
    assert body["suggested_candidate_id"] is None
    assert len(body["discovery_candidates"]) == 1
    assert body["discovery_candidates"][0]["hook"].startswith("Genetic uniformity")
    assert body["discovery_candidates"][0]["evidence"][0]["url"] == "https://example.org/blight"


def test_discovery_candidates_collapse_paraphrased_duplicates(client: TestClient) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]

    class DuplicateCandidateAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(question="Which bred crop became dangerous?", why_promising="A withdrawal creates records.", seed_detail="Selective breeding.", discovery_target="A crop withdrawn for a chemical hazard.", search_queries=["bred crop withdrawn chemical hazard"]),
                DiscoveryDirection(question="When did crop uniformity amplify risk?", why_promising="A disease outcome may be documented.", seed_detail="Ecological vulnerability.", discovery_target="A named disease event linked to crop uniformity.", search_queries=["crop uniformity disease vulnerability case"]),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="The potato bred into a safety warning",
                discovery="The Lenape potato was withdrawn for dangerously elevated solanine.",
                why_surprising="Breeding a commercial crop produced a chemical hazard.", connection_to_library=None,
                what_remains_uncertain="The causal parent line is not established.", next_rabbit_hole="Trace the parent lines.",
                claims=[ResearchedDiscoveryClaim(claim_type="supported_fact", text="Lenape was withdrawn for elevated solanine.", source_ids=["discovery-source-1"])],
                considered_findings=[
                    considered_discovery_finding("The Lenape potato was withdrawn for dangerously elevated solanine.", ["discovery-source-1"]),
                    considered_discovery_finding("Dangerously elevated solanine: the Lenape potato was withdrawn.", ["discovery-source-1"], selected=False),
                    considered_discovery_finding("Genetic uniformity left potato crops vulnerable to blight.", ["discovery-source-2"], selected=False, direction_index=1),
                ],
                selection_rationale="The Lenape withdrawal was the strongest particular.",
            )

    class DuplicateResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Two supported findings.", sources=[
                RetrievedSource(url="https://example.org/lenape", title="Lenape case", excerpt="Lenape was withdrawn for elevated solanine."),
                RetrievedSource(url="https://example.org/blight", title="Blight case", excerpt="Genetic uniformity increased vulnerability to blight."),
            ])

    app.dependency_overrides[get_ai_client] = lambda: DuplicateCandidateAI()
    app.dependency_overrides[get_research_provider] = lambda: DuplicateResearch()
    response = client.post("/api/discoveries/research", json={"item_id": item_id})

    assert response.status_code == 200
    candidates = response.json()["discovery_candidates"]
    assert len(candidates) == 2
    assert sum(candidate["ai_suggested"] for candidate in candidates) == 1


def test_discovery_diagnostics_are_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("DISCOVERY_DEBUG", raising=False)
    caplog.set_level(logging.INFO, logger="app.main")

    class DiagnosticAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="What did the instrument preserve?", why_promising="A physical trace may survive.",
                    seed_detail="Remembered weather.", discovery_target="A named instrument and its surviving trace.",
                    search_queries=["museum instrument weather trace"],
                ),
                DiscoveryDirection(
                    question="Which observation became an object?", why_promising="A record may become material.",
                    seed_detail="Memory made physical.", discovery_target="A specific record embodied in a physical mechanism.",
                    search_queries=["physical mechanism records weather history"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=True, rejection_reason="", title="A recorded trace",
                discovery="A museum instrument preserved a weather trace.", why_surprising="The observation became physical.",
                connection_to_library=None, what_remains_uncertain="The excerpt covers one instrument.",
                next_rabbit_hole="Inspect the museum catalog entry.", claims=[ResearchedDiscoveryClaim(
                    claim_type="supported_fact", text="The instrument preserved a trace.",
                    source_ids=["discovery-source-1"],
                )],
                considered_findings=[considered_discovery_finding("A museum instrument preserved a weather trace.")],
                selection_rationale="The physical trace was the strongest documented particular.",
            )

    class DiagnosticResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="One instrument was documented.", sources=[RetrievedSource(
                url="https://example.org/instrument", title="Instrument catalog",
                excerpt="The instrument preserved a trace.",
            )])

    run_researched_discovery(
        {"id": 1, "title": "A clock that remembers weather", "note": "private note", "concepts": []},
        DiagnosticAI(), DiagnosticResearch(),
    )

    assert "discovery_debug" not in caplog.text


def test_discovery_diagnostics_capture_stages_and_redact_sensitive_fields(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DISCOVERY_DEBUG", "1")
    caplog.set_level(logging.INFO, logger="app.main")
    exposed_key = "sk-" + "or-v1-" + "EXAMPLESECRET123456"
    private_note = "PRIVATE NOTE MUST NOT APPEAR"

    class DiagnosticAI(FakeAI):
        def propose_discovery_directions(self, seed) -> DiscoveryDirectionBatch:
            return DiscoveryDirectionBatch(directions=[
                DiscoveryDirection(
                    question="Which archive records the contradiction?", why_promising=f"Authorization: Bearer {exposed_key}",
                    seed_detail="A concrete contradiction.", discovery_target="A named archival incident and its outcome.",
                    search_queries=[f"archive contradiction api_key={exposed_key}"],
                ),
                DiscoveryDirection(
                    question="What result overturned the expectation?", why_promising="A result may survive in a report.",
                    seed_detail="An overturned expectation.", discovery_target="A documented result that contradicted the researchers' expectation.",
                    search_queries=["documented unexpected experimental result"],
                ),
            ])

        def synthesize_researched_discovery(self, seed, directions, research_synthesis, sources) -> ResearchedDiscoveryDraft:
            return ResearchedDiscoveryDraft(
                accepted=False, rejection_reason="Only a broad overview was retrieved.", title="No finding",
                discovery="No distinctive result was established.", why_surprising="",
                connection_to_library=f"api_key={exposed_key}", what_remains_uncertain="The particular is missing.",
                next_rabbit_hole="Search the named archive.", claims=[ResearchedDiscoveryClaim(
                    claim_type="interpretation", text="The overview hints at a contradiction.", source_ids=[],
                )],
                considered_findings=[considered_discovery_finding(
                    "The overview hints at a contradiction.", selected=False,
                )],
                selection_rationale=f"No candidate had direct support. api_key={exposed_key}",
            )

    class DiagnosticResearch:
        def research_discovery(self, seed, directions) -> ResearchPacket:
            return ResearchPacket(synthesis="Broad background only.", sources=[RetrievedSource(
                url="https://user:password@example.org/archive/item?token=secret#private",
                title="Archive overview", excerpt=("bounded evidence " * 45) + exposed_key + " TAIL_SENTINEL",
            )])

    with pytest.raises(NoDiscoveryError, match="broad overview"):
        run_researched_discovery(
            {
                "id": 1, "title": "Diagnostic seed", "note": private_note,
                "concepts": [{"name": f"A concept containing {exposed_key}"}],
            },
            DiagnosticAI(), DiagnosticResearch(),
        )

    output = caplog.text
    for stage in ("seed", "directions", "research_request", "sources", "synthesis", "outcome"):
        assert f'"stage": "{stage}"' in output
    assert '"submitted_search_queries"' in output
    assert '"provider_executed_queries": null' in output
    assert '"considered_findings"' in output
    assert '"selection_rationale"' in output
    assert '"outcome": "no_discovery"' in output
    assert "Only a broad overview was retrieved." in output
    assert "https://example.org/archive/item" in output
    assert "[redacted]" in output
    assert exposed_key not in output
    assert private_note not in output
    assert "user:password" not in output
    assert "token=secret" not in output
    assert "TAIL_SENTINEL" not in output


def test_sidequest_allows_two_items_of_the_same_type(client: TestClient) -> None:
    first = create_item(client, title="Distributed systems").json()["id"]
    second = create_item(client, title="The Overstory").json()["id"]
    for item_id, name in [(first, "Fault tolerance"), (second, "Ecological resilience")]:
        client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"})

    class SameTypeLibraryWinnerAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            external = super().generate_candidates(contexts).candidates
            context = contexts[0]
            clue = context["item_a"]
            lead = context["possible_library_leads"][0]
            library = external[0].model_copy(update={
                "item_a_id": clue["id"], "item_b_id": lead["id"],
                "origin_item_ids": [clue["id"], lead["id"]],
                "title": "Failure as a source of resilience",
                "lead": "The Overstory's ecological resilience as a library connection",
                "lead_kind": "library_item", "lead_category": "library_curiosity",
                "connection": "Fault tolerance and ecological resilience both preserve function by distributing recovery across partially independent components.",
                "bridge": "Distributed recovery through partially independent components",
                "new_question": "Which forms of redundancy preserve recovery capacity without making every component interchangeable?",
            })
            return CandidateBatch(candidates=[*external, library])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            selected_index = next(index for index, candidate in enumerate(candidates) if candidate["lead_kind"] == "library_item")
            selected = candidates[selected_index]
            return CritiquedConnection(
                accepted=True, rejection_reason="", **selected,
                evidence_status="Exploratory.", concepts_used=selected["clue_concept_names"], evidence=[],
                selected_candidate_index=selected_index,
                candidate_assessments=[SidequestCandidateAssessment(
                    candidate_index=index, external_lead_aligned=True, mechanism_explained=True,
                    next_investigation_follows_lead=True, valid=True, rejection_reason="",
                ) for index, _candidate in enumerate(candidates)],
            )

    app.dependency_overrides[get_ai_client] = lambda: SameTypeLibraryWinnerAI()
    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert {created.json()["item_a_id"], created.json()["item_b_id"]} == {first, second}


def test_sidequest_can_open_a_deduplicated_direction_from_one_random_thought(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    thought_id = create_item(client, kind="thought", title="What if maps remembered where we hesitated?").json()["id"]
    client.post(f"/api/items/{thought_id}/concepts", json={"name": "Hesitation as spatial data", "category": "mechanism"})

    class SingleDirectionAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            single = next(context for context in contexts if context["mode"] == "clue_investigation")
            item_id = single["item_a"]["id"]  # type: ignore[index]
            counterfactual = ConnectionCandidate(
                item_a_id=item_id, item_b_id=item_id, origin_item_ids=[item_id],
                discovery_kind="creative_impulse", title="A map of almost-chosen paths",
                invitation="Trace the places where uncertainty changed your route.",
                connection="Recording hesitation could turn a route map into a portrait of decisions that nearly happened.",
                bridge="Spatial traces of unrealized decisions", why_interesting="It treats uncertainty as information rather than noise.",
                where_it_breaks="A pause can reflect distraction, access barriers, or sensor error rather than meaningful uncertainty.",
                new_question="What would a map reveal if it preserved abandoned routes as carefully as completed ones?",
                clue="The thought asks what changes when a map records hesitation instead of only completed movement.",
                clue_item_id=item_id, clue_concept_names=["Hesitation as spatial data"],
                lead="Counterfactual cartography: mapping routes that were considered but not taken.",
                lead_kind="unfamiliar_subject", lead_category="artistic_technique", reasoning_steps=[
                    "A hesitation marks a moment when more than one route still felt possible.",
                    "Preserving those moments turns a route record into a map of unrealized decisions.",
                ], claims=[
                    SidequestClaim(claim_type="interpretation", text="Hesitation can be treated as information rather than discarded as noise."),
                    SidequestClaim(claim_type="speculative_hypothesis", text="A counterfactual map might reveal how uncertainty shaped a journey."),
                ],
            )
            desire_path = counterfactual.model_copy(update={
                "title": "Where indecision leaves a path",
                "lead": "Desire paths as records of repeated route disagreement",
                "lead_category": "phenomenon",
                "connection": "Hesitation marks could be compared with physical traces that accumulate when intended and preferred routes diverge.",
                "bridge": "Accumulated route choices making disagreement with a planned path visible",
                "new_question": "Where do desire paths reveal repeated uncertainty rather than a stable preferred shortcut?",
                "reasoning_steps": [
                    "A hesitation records friction between the route offered and the route still being considered.",
                    "Repeated route friction can accumulate into a visible alternative path through a landscape.",
                ],
            })
            return CandidateBatch(candidates=[counterfactual, desire_path])

    app.dependency_overrides[get_ai_client] = lambda: SingleDirectionAI()
    with caplog.at_level("INFO", logger="app.main"):
        created = client.post("/api/sidequests/surprise")
    repeated = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert created.json()["origin_item_ids"] == [thought_id]
    assert created.json()["origin_titles"] == ["What if maps remembered where we hesitated?"]
    assert created.json()["discovery_kind"] == "creative_impulse"
    assert created.json()["invitation"] == "Trace the places where uncertainty changed your route."
    assert created.json()["detective_version"] == 1
    assert created.json()["lead"]["kind"] == "unfamiliar_subject"
    assert {claim["claim_type"] for claim in created.json()["claims"]} == {"interpretation", "speculative_hypothesis"}
    assert created.json()["evidence"] == []
    assert '"clue_dossiers": 1' in caplog.text
    assert '"candidate_contexts": 1' in caplog.text
    assert '"generated_candidates": 2' in caplog.text
    assert '"sent_to_critic": 2' in caplog.text
    assert '"outcome": "saved"' in caplog.text
    assert repeated.status_code == 422
    assert "already been explored" in repeated.json()["detail"]


def test_detective_mode_follows_a_sourced_clue_across_three_curiosity_types(client: TestClient) -> None:
    anchors = {
        "Piranesi": [("system", "Tides as external memory", "Piranesi records the House's tides in journals so recurring floods become navigable rather than unknowable.")],
        "Articles of Interest: Pockets": [("system", "Carrying capacity hidden in clothing", "Pocket dimensions determine which everyday objects a wearer can privately carry while keeping both hands free.")],
    }

    class DossierResearch(FakeResearch):
        def research(self, item: dict[str, object]) -> ResearchPacket:
            detail = anchors[str(item["title"])][0][2]
            source = RetrievedSource(url=f"https://example.com/{str(item['title']).lower().replace(' ', '-')}", title=f"{item['title']} source", excerpt=detail)
            return ResearchPacket(synthesis="A source identifies the work and a concrete clue.", sources=[source, source])

        def research_problems(self, _item: dict[str, object], _problems: list[dict[str, object]]) -> ResearchPacket:
            raise AssertionError("These work-supported clues do not need external research.")

    class DetectiveAI(WorkFirstDiscoveryAI):
        anchors_by_title = anchors
        seen_kinds: set[str] = set()
        last_contexts: list[dict[str, object]] = []
        critic_received_dossiers = False

        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            self.last_contexts = contexts
            self.seen_kinds = {
                str(item["kind"])
                for context in contexts
                for item in (context["item_a"], *context["possible_library_leads"])
                if item is not None
            }
            piranesi_thought = next(
                context for context in contexts
                if context["item_a"]["title"] == "Piranesi"
            )
            piranesi = piranesi_thought["item_a"]
            thought = next(item for item in piranesi_thought["possible_library_leads"] if item["title"] == "What if maps remembered where we hesitated?")
            clue_concept = piranesi["concepts"][0]
            source_id = clue_concept["sources"][0]["source_id"]
            good = ConnectionCandidate(
                item_a_id=piranesi["id"], item_b_id=thought["id"], origin_item_ids=[piranesi["id"], thought["id"]],
                title="The map that remembers uncertainty", clue="Piranesi's tidal journals turn recurring floods into a navigable external memory.",
                clue_item_id=piranesi["id"], clue_concept_names=[clue_concept["name"]], clue_source_ids=[source_id],
                lead="The saved thought suggests counterfactual cartography: maps that preserve moments when another route was still possible.",
                lead_kind="library_item", lead_category="library_curiosity", reasoning_steps=[
                    "The tidal journal makes a changing environment legible by recording recurrence outside the mind.",
                    "A hesitation map could do the same for decisions, preserving the moment before one possible route disappeared.",
                    "That turns a map from a record of movement into a record of uncertainty.",
                ], connection="External records may make unstable environments navigable—and might do the same for unstable decisions.",
                bridge="External memory for patterns that would otherwise vanish", why_interesting="A practical survival record becomes a strange model for remembering paths not taken.",
                where_it_breaks="Tides recur independently of an observer, while hesitation is subjective and may reflect distraction rather than a meaningful choice.",
                new_question="What would become visible if maps recorded the instant a route stopped being one possibility among several?",
                claims=[
                    SidequestClaim(claim_type="supported_fact", text="Piranesi records recurring tides in journals to navigate the House.", source_ids=[source_id]),
                    SidequestClaim(claim_type="interpretation", text="The journals function as memory outside the individual mind."),
                    SidequestClaim(claim_type="speculative_hypothesis", text="A map of hesitation might preserve unrealized decisions as spatial information."),
                ], invitation="Look for the moment a possibility disappears.", discovery_kind="new_lens",
            )
            podcast = next(context for context in contexts if context["item_a"]["title"] == "Articles of Interest: Pockets")
            podcast_lead = podcast["possible_library_leads"][0]
            generic = ConnectionCandidate(
                item_a_id=podcast["item_a"]["id"], item_b_id=podcast_lead["id"], origin_item_ids=[podcast["item_a"]["id"], podcast_lead["id"]],
                title="Two stories about design", connection="Both entries explore human nature through design.", bridge="A broad theme of design and humanity",
                why_interesting="They seem related.", where_it_breaks="Their particular mechanisms and contexts are different enough that this broad comparison explains very little.",
                new_question="What else is similar?",
            )
            external_memory = good.model_copy(update={
                "item_b_id": piranesi["id"], "origin_item_ids": [piranesi["id"]],
                "title": "When a notebook becomes part of navigation",
                "lead": "Cognitive offloading in changing environments", "lead_kind": "unfamiliar_subject",
                "lead_category": "scientific_mechanism",
                "connection": "The tidal journal suggests treating an external record as part of the system that perceives and navigates change.",
                "bridge": "Offloading recurring environmental patterns into an external memory system",
                "new_question": "Which navigation practices fail when their external records are removed at the moment conditions begin to change?",
                "claims": [SidequestClaim(claim_type="interpretation", text="The journal can be read as external memory for a changing environment.")],
            })
            archive = good.model_copy(update={
                "item_b_id": piranesi["id"], "origin_item_ids": [piranesi["id"]],
                "title": "An archive that predicts recurrence",
                "lead": "Phenological calendars that record recurring environmental events", "lead_kind": "unfamiliar_subject",
                "lead_category": "practice",
                "connection": "The tidal journal opens an investigation into records organized around recurrence rather than chronological completeness.",
                "bridge": "Using repeated observations to anticipate the return of environmental conditions",
                "new_question": "How do seasonal calendars decide which recurring event is reliable enough to guide a future action?",
                "claims": [SidequestClaim(claim_type="speculative_hypothesis", text="Recurrence-based environmental calendars may be a useful outside lead for the tidal journals.")],
            })
            return CandidateBatch(candidates=[good, external_memory, archive, generic])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            self.critic_received_dossiers = bool(dossiers and dossiers[0].get("item_a"))
            candidate = candidates[0]
            return CritiquedConnection(
                accepted=True, rejection_reason="", **candidate, evidence_status="Exploratory.",
                concepts_used=candidate["clue_concept_names"], evidence=[], selected_candidate_index=0,
                candidate_assessments=[SidequestCandidateAssessment(
                    candidate_index=index, external_lead_aligned=True, mechanism_explained=True,
                    next_investigation_follows_lead=True, valid=True, rejection_reason="",
                ) for index, _candidate in enumerate(candidates)],
            )

    ai = DetectiveAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: DossierResearch()
    piranesi = create_item(client, title="Piranesi").json()["id"]
    pockets = create_item(client, kind="podcast", title="Articles of Interest: Pockets").json()["id"]
    thought = create_item(client, kind="thought", title="What if maps remembered where we hesitated?").json()["id"]
    assert client.post(f"/api/items/{piranesi}/enrich").status_code == 200
    assert client.post(f"/api/items/{pockets}/enrich").status_code == 200
    assert client.post(f"/api/items/{thought}/concepts", json={"name": "Hesitation as spatial data", "category": "mechanism"}).status_code == 201

    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    body = created.json()
    assert ai.seen_kinds == {"book", "podcast", "thought"}
    assert ai.critic_received_dossiers is True
    assert body["clue"]["concept_names"] == ["Tides as external memory"]
    assert body["lead"]["kind"] == "library_item"
    assert body["lead"]["category"] == "library_curiosity"
    assert all(context["mode"] == "clue_investigation" and context["item_b"] is None for context in ai.last_contexts)
    assert len(body["reasoning_steps"]) == 3
    assert {claim["claim_type"] for claim in body["claims"]} == {"supported_fact", "interpretation", "speculative_hypothesis"}
    assert body["evidence"] == ["https://example.com/piranesi"]
    assert "sourced details supported" in body["evidence_status"]


def test_detective_mode_rejects_the_psalm_sherlock_abstract_parallel(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    psalm = create_item(client, title="A Psalm for the Wild-Built").json()["id"]
    sherlock = create_item(client, title="Sherlock Holmes").json()["id"]
    client.post(f"/api/items/{psalm}/concepts", json={"name": "Ecological restraint as infrastructure", "category": "mechanism"})
    client.post(f"/api/items/{sherlock}/concepts", json={"name": "Observation as method", "category": "mechanism"})

    class AbstractParallelAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            context = next(entry for entry in contexts if entry["item_a"]["id"] == psalm)
            lead = next(item for item in context["possible_library_leads"] if item["id"] == sherlock)
            return CandidateBatch(candidates=[ConnectionCandidate(
                item_a_id=psalm, item_b_id=lead["id"], origin_item_ids=[psalm, sherlock],
                title="Exploring the Nature of Independence",
                clue="Humans adopt a sustainable lifestyle with less dependence on advanced technology.",
                clue_item_id=psalm, clue_concept_names=["Ecological restraint as infrastructure"],
                lead="Independence and relationships with technology in A Psalm for the Wild-Built and Sherlock Holmes.",
                lead_kind="library_item", lead_category="library_curiosity",
                reasoning_steps=[
                    "Both works depict social change.",
                    "One involves self-sufficiency while the other prizes reason and observation.",
                ],
                connection="Both narratives consider independence through their characters' relationship to society and technology.",
                bridge="The broad theme of personal independence", why_interesting="It compares two approaches to autonomy.",
                where_it_breaks="The connection relies on abstract parallels rather than a shared causal or structural mechanism.",
                new_question="What ethical dilemmas arise around independence?",
                claims=[SidequestClaim(claim_type="interpretation", text="Both stories can be read as being about independence.")],
            )])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            raise AssertionError("The deterministic gate should reject the self-disqualifying abstract parallel before critique.")

    app.dependency_overrides[get_ai_client] = lambda: AbstractParallelAI()
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 422
    assert "No compelling case yet" in response.json()["detail"]
    assert "abstract or thematic parallel" in response.json()["detail"]
    assert '"rejected_candidates": 1' in caplog.text
    assert '"sent_to_critic": 0' in caplog.text
    assert "abstract or thematic parallel" in caplog.text
    assert '"outcome": "all_candidates_rejected"' in caplog.text
    assert client.get("/api/sidequests").json() == []


def test_detective_mode_compares_outward_leads_before_a_same_author_library_match(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    hail_mary = create_item(client, title="Project Hail Mary", creator="Andy Weir").json()["id"]
    martian = create_item(client, title="The Martian", creator="Andy Weir").json()["id"]
    client.post(f"/api/items/{hail_mary}/concepts", json={"name": "Selective breeding for interspecies cooperation", "category": "mechanism"})
    client.post(f"/api/items/{martian}/concepts", json={"name": "Agriculture under extreme scarcity", "category": "mechanism"})

    class OutwardCompetitionAI(FakeAI):
        generated_kinds: list[str] = []
        candidate_kinds: list[str] = []
        candidate_categories: list[str] = []

        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            context = next(entry for entry in contexts if entry["item_a"]["id"] == hail_mary)
            library_lead = next(item for item in context["possible_library_leads"] if item["id"] == martian)
            assert "creator" not in library_lead
            common = {
                "clue": "A constrained breeding program selects Taumoeba that can address the Astrophage threat.",
                "clue_item_id": hail_mary,
                "clue_concept_names": ["Selective breeding for interspecies cooperation"],
            }
            weak_library = ConnectionCandidate(
                item_a_id=hail_mary, item_b_id=martian, origin_item_ids=[hail_mary, martian],
                title="Two survival toolkits", lead="The Martian's agricultural survival methods.",
                lead_kind="library_item", lead_category="library_curiosity",
                reasoning_steps=[
                    "The Taumoeba program changes a living population under survival pressure.",
                    "The Martian uses cultivation to produce food under a different survival pressure.",
                ],
                claims=[SidequestClaim(claim_type="interpretation", text="The two works use different biological interventions as survival tools.")],
                connection="Both works explore survival through biological problem-solving.",
                bridge="Intervening in living systems under resource scarcity",
                why_interesting="It places two survival techniques beside each other.",
                where_it_breaks="Agricultural cultivation does not establish a mechanism for interspecies cooperation or selective breeding.",
                new_question="How can biology solve survival problems?", **common,
            )
            outward_science = ConnectionCandidate(
                item_a_id=hail_mary, item_b_id=hail_mary, origin_item_ids=[hail_mary],
                title="Making evolution choose the less dangerous exit", lead="Evolutionary steering in antimicrobial resistance.",
                lead_kind="unfamiliar_subject", lead_category="scientific_mechanism",
                reasoning_steps=[
                    "The breeding program turns variation across generations into a search through possible traits.",
                    "Evolutionary steering asks whether the selective environment can make one adaptive route easier than another.",
                    "The shared mechanism is intervention in the fitness landscape, not merely the use of biology during a crisis.",
                ],
                claims=[
                    SidequestClaim(claim_type="interpretation", text="The fictional breeding sequence can be read as a constrained search through traits."),
                    SidequestClaim(claim_type="speculative_hypothesis", text="Evolutionary steering may be a useful outside lens for asking which adaptive escape routes an intervention rewards."),
                ],
                connection="The clue suggests treating selection pressure itself as the control surface rather than treating adaptation as an obstacle.",
                bridge="Reshaping a fitness landscape to redirect adaptation",
                why_interesting="It turns a creature-breeding solution into a question about whether evolution's escape routes can be designed.",
                where_it_breaks="A fictional breeding program with a known target is much simpler than steering real populations with uncertain mutations and ecological effects.",
                new_question="Could a treatment make a pathogen's easiest evolutionary escape route also make it less dangerous?", **common,
            )
            outward_art = ConnectionCandidate(
                item_a_id=hail_mary, item_b_id=hail_mary, origin_item_ids=[hail_mary],
                title="A montage made out of generations", lead="The breeding montage as an artistic technique for making iteration visible.",
                lead_kind="unfamiliar_subject", lead_category="artistic_technique",
                reasoning_steps=[
                    "Each generation makes a hidden process of variation and selection observable through changed outcomes.",
                    "A montage can compress repeated trials while preserving the causal rhythm of failure, adjustment, and inheritance.",
                ],
                claims=[SidequestClaim(claim_type="interpretation", text="The breeding sequence can function as a narrative compression of iterative learning.")],
                connection="Repeated generations give narrative form to a process whose mechanism would otherwise remain invisible.",
                bridge="Compressing iterative change while preserving causal steps",
                why_interesting="The scientific procedure doubles as a storytelling device for making slow change legible.",
                where_it_breaks="Narrative compression can imply a cleaner path of progress than an actual breeding process would produce.",
                new_question="How do films and novels compress repeated failure without erasing the moment when the method changes?", **common,
            )
            generated = [weak_library, outward_science, outward_art]
            self.generated_kinds = [candidate.lead_kind for candidate in generated]
            return CandidateBatch(candidates=generated)

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            self.candidate_kinds = [str(candidate["lead_kind"]) for candidate in candidates]
            self.candidate_categories = [str(candidate["lead_category"]) for candidate in candidates]
            selected = next(candidate for candidate in candidates if candidate["lead_category"] == "scientific_mechanism")
            selected_index = candidates.index(selected)
            return CritiquedConnection(
                accepted=True, rejection_reason="", **selected,
                evidence_status="Exploratory · open hypothesis; not externally verified.",
                concepts_used=selected["clue_concept_names"], evidence=[],
                selected_candidate_index=selected_index,
                candidate_assessments=[SidequestCandidateAssessment(
                    candidate_index=index, external_lead_aligned=True, mechanism_explained=True,
                    next_investigation_follows_lead=True, valid=True, rejection_reason="",
                ) for index, _candidate in enumerate(candidates)],
            )

    ai = OutwardCompetitionAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 201
    assert ai.generated_kinds.count("unfamiliar_subject") == 2
    assert ai.generated_kinds.count("library_item") == 1
    assert ai.candidate_kinds.count("unfamiliar_subject") == 2
    assert ai.candidate_kinds.count("library_item") == 0
    assert {"scientific_mechanism", "artistic_technique"} == set(ai.candidate_categories)
    assert response.json()["origin_item_ids"] == [hail_mary]
    assert response.json()["lead"]["category"] == "scientific_mechanism"
    assert "pathogen's easiest evolutionary escape route" in response.json()["new_question"]
    assert response.json()["evidence"] == []
    assert "at least two rival" in CONNECTION_SYSTEM_PROMPT
    assert "at most one library-item candidate" in CONNECTION_SYSTEM_PROMPT
    assert "author, franchise, genre" in CONNECTION_SYSTEM_PROMPT
    assert '"generated_candidates": 3' in caplog.text
    assert '"rejected_candidates": 1' in caplog.text
    assert '"sent_to_critic": 2' in caplog.text
    assert "broad shared topic or format comparison" in caplog.text


def test_detective_mode_rejects_the_single_laundered_hydroponics_candidate_before_critique(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    hail_mary = create_item(client, title="Project Hail Mary", creator="Andy Weir").json()["id"]
    client.post(f"/api/items/{hail_mary}/concepts", json={"name": "Selective breeding for interspecies cooperation", "category": "mechanism"})

    class LaunderedHydroponicsAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            return CandidateBatch(candidates=[ConnectionCandidate(
                item_a_id=hail_mary, item_b_id=hail_mary, origin_item_ids=[hail_mary],
                title="Growing a way home", clue="Taumoeba are selectively bred to address the Astrophage threat.",
                clue_item_id=hail_mary, clue_concept_names=["Selective breeding for interspecies cooperation"],
                lead="Hydroponic Farming in Space", lead_kind="unfamiliar_subject", lead_category="scientific_mechanism",
                reasoning_steps=[
                    "Project Hail Mary uses biological problem-solving during a survival crisis.",
                    "The Martian uses agriculture and innovative problem-solving during a survival crisis.",
                ],
                connection="The novels place biological ingenuity beside survival under extreme conditions.",
                bridge="Biological problem-solving under extreme survival constraints",
                why_interesting="The comparison highlights ingenuity in hostile environments.",
                where_it_breaks="Hydroponic cultivation does not explain selective breeding or interspecies cooperation.",
                new_question="How could interspecies cooperation help solve real-world survival problems?",
                claims=[SidequestClaim(claim_type="interpretation", text="The survival methods can be compared as biological ingenuity.")],
            )])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            raise AssertionError("A one-candidate external batch must not reach the critic.")

    app.dependency_overrides[get_ai_client] = lambda: LaunderedHydroponicsAI()
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 422
    assert "two distinct external investigation directions" in response.json()["detail"]
    assert '"outcome": "candidate_count_shortfall"' in caplog.text
    assert '"sent_to_critic": 0' in caplog.text
    assert '"outcome": "insufficient_external_competition"' in caplog.text
    assert client.get("/api/sidequests").json() == []


def test_detective_mode_rejects_external_lead_laundering_after_semantic_critique(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    hail_mary = create_item(client, title="Project Hail Mary", creator="Andy Weir").json()["id"]
    client.post(f"/api/items/{hail_mary}/concepts", json={"name": "Selective breeding for interspecies cooperation", "category": "mechanism"})
    common = {
        "item_a_id": hail_mary, "item_b_id": hail_mary, "origin_item_ids": [hail_mary],
        "clue": "Taumoeba are selectively bred to address the Astrophage threat.", "clue_item_id": hail_mary,
        "clue_concept_names": ["Selective breeding for interspecies cooperation"], "lead_kind": "unfamiliar_subject",
    }
    laundered = ConnectionCandidate(
        **common, title="Growing a way home", lead="Hydroponic Farming in Space", lead_category="scientific_mechanism",
        reasoning_steps=["Project Hail Mary uses biological problem-solving during a survival crisis.", "The Martian uses agriculture during a survival crisis."],
        connection="The novels place biological ingenuity beside survival under extreme conditions.",
        bridge="Biological problem-solving under extreme survival constraints", why_interesting="It compares two survival toolkits.",
        where_it_breaks="Hydroponic cultivation does not explain selective breeding or interspecies cooperation.",
        new_question="How could interspecies cooperation help solve real-world survival problems?",
        claims=[SidequestClaim(claim_type="interpretation", text="The survival methods can be compared as biological ingenuity.")],
    )
    aligned = ConnectionCandidate(
        **common, title="Making evolution choose the less dangerous exit", lead="Evolutionary steering in antimicrobial resistance",
        lead_category="scientific_mechanism",
        reasoning_steps=["Breeding turns variation into a constrained search through traits.", "Evolutionary steering asks which adaptive route a selective environment makes easiest."],
        connection="The clue suggests treating selection pressure as an intervention surface.",
        bridge="Reshaping a fitness landscape to redirect adaptation", why_interesting="It asks whether an escape route can be designed.",
        where_it_breaks="A fictional breeding target is simpler than adaptation in a changing real population.",
        new_question="Could a treatment make a pathogen's easiest evolutionary escape route also make it less dangerous?",
        claims=[SidequestClaim(claim_type="speculative_hypothesis", text="Evolutionary steering may be a useful outside lens for the breeding sequence.")],
    )

    class SemanticLaunderingAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            return CandidateBatch(candidates=[laundered, aligned])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            return CritiquedConnection(
                accepted=True, rejection_reason="", **candidates[0], evidence_status="Exploratory.",
                concepts_used=candidates[0]["clue_concept_names"], evidence=[], selected_candidate_index=0,
                candidate_assessments=[
                    SidequestCandidateAssessment(
                        candidate_index=0, external_lead_aligned=False, mechanism_explained=False,
                        next_investigation_follows_lead=False, valid=False,
                        rejection_reason="The reasoning compares library works and never investigates hydroponics.",
                    ),
                    SidequestCandidateAssessment(
                        candidate_index=1, external_lead_aligned=True, mechanism_explained=True,
                        next_investigation_follows_lead=True, valid=True, rejection_reason="",
                    ),
                ],
            )

    app.dependency_overrides[get_ai_client] = lambda: SemanticLaunderingAI()
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 422
    assert "Fewer than two external directions remained aligned" in response.json()["detail"]
    assert '"external_lead_mismatch": 1' in caplog.text
    assert '"mechanism_missing": 1' in caplog.text
    assert '"next_investigation_mismatch": 1' in caplog.text
    assert '"outcome": "external_lead_mismatch"' in caplog.text
    assert client.get("/api/sidequests").json() == []


def test_detective_mode_collapses_duplicate_external_directions_before_critique(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    item_id = create_item(client, title="Project Hail Mary").json()["id"]
    concept_name = "Selective breeding for interspecies cooperation"
    client.post(f"/api/items/{item_id}/concepts", json={"name": concept_name, "category": "mechanism"})
    common = {
        "item_a_id": item_id, "item_b_id": item_id, "origin_item_ids": [item_id],
        "clue": "Taumoeba are selectively bred under a survival constraint.", "clue_item_id": item_id,
        "clue_concept_names": [concept_name], "lead_kind": "unfamiliar_subject",
    }
    steering = ConnectionCandidate(
        **common, title="Choosing evolution's exit", lead="Evolutionary steering in antimicrobial resistance",
        lead_category="scientific_mechanism", connection="Selection pressure can reshape which adaptive escape route is easiest.",
        bridge="Reshaping a fitness landscape to redirect adaptation", why_interesting="The intervention targets evolution's available exits.",
        where_it_breaks="A fictional controlled breeding sequence is simpler than a changing clinical population.",
        new_question="Could a treatment make a pathogen's easiest evolutionary escape route less dangerous?",
        reasoning_steps=["Breeding searches variation by repeatedly selecting traits.", "Evolutionary steering changes which adaptive route selection rewards."],
        claims=[SidequestClaim(claim_type="speculative_hypothesis", text="Evolutionary steering may be a useful outside lens.")],
    )
    steering_paraphrase = steering.model_copy(update={
        "title": "Steering an evolutionary escape",
        "lead": "Antimicrobial-resistance evolutionary steering",
    })
    cultural_selection = ConnectionCandidate(
        **common, title="When selection becomes culture", lead="Cultural selection in craft apprenticeship",
        lead_category="practice", connection="Repeated choices about which variations to preserve can shape a craft tradition.",
        bridge="Selective preservation accumulating into a transmitted practice", why_interesting="Selection need not be genetic to redirect variation over time.",
        where_it_breaks="Craft choices are intentional and socially transmitted rather than inherited biologically.",
        new_question="Which workshop traditions changed because apprentices repeatedly preserved one difficult variation?",
        reasoning_steps=["The clue repeatedly preserves selected variations.", "Craft apprenticeship can likewise preserve chosen variations through teaching."],
        claims=[SidequestClaim(claim_type="speculative_hypothesis", text="Craft transmission may offer a non-biological comparison for cumulative selection.")],
    )

    class DuplicateDirectionAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            return CandidateBatch(candidates=[steering, steering_paraphrase, cultural_selection])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            assert len(candidates) == 2
            return CritiquedConnection(
                accepted=True, rejection_reason="", **candidates[0], evidence_status="Exploratory.",
                concepts_used=candidates[0]["clue_concept_names"], evidence=[], selected_candidate_index=0,
                candidate_assessments=[SidequestCandidateAssessment(
                    candidate_index=index, external_lead_aligned=True, mechanism_explained=True,
                    next_investigation_follows_lead=True, valid=True, rejection_reason="",
                ) for index in range(2)],
            )

    app.dependency_overrides[get_ai_client] = lambda: DuplicateDirectionAI()
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 201
    assert '"duplicate_external_leads": 1' in caplog.text
    assert '"distinct_external_candidates": 2' in caplog.text
    assert '"sent_to_critic": 2' in caplog.text


def test_detective_mode_reports_when_generation_returns_no_candidates(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    item_id = create_item(client, kind="thought", title="A clock that forgets noon").json()["id"]
    client.post(f"/api/items/{item_id}/concepts", json={"name": "Selective timekeeping", "category": "mechanism"})

    class NoCandidateAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            return CandidateBatch(candidates=[])

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            raise AssertionError("An empty generation result must not invoke the critic.")

    app.dependency_overrides[get_ai_client] = lambda: NoCandidateAI()
    with caplog.at_level("INFO", logger="app.main"):
        response = client.post("/api/sidequests/surprise")

    assert response.status_code == 422
    assert "No compelling case yet" in response.json()["detail"]
    assert '"generated_candidates": 0' in caplog.text
    assert '"sent_to_critic": 0' in caplog.text
    assert '"outcome": "no_generated_candidates"' in caplog.text
    assert client.get("/api/sidequests").json() == []


def test_detective_mode_rejects_an_invalid_evidence_reference(client: TestClient) -> None:
    first = create_item(client, title="First clue").json()["id"]
    second = create_item(client, kind="thought", title="Second clue").json()["id"]
    client.post(f"/api/items/{first}/concepts", json={"name": "Observed pattern", "category": "mechanism"})
    client.post(f"/api/items/{second}/concepts", json={"name": "Possible lead", "category": "personal"})

    class FabricatedSourceAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            candidate = super().generate_candidates(contexts).candidates[0]
            return CandidateBatch(candidates=[candidate.model_copy(update={
                "clue_source_ids": ["source-fabricated"],
                "claims": [SidequestClaim(claim_type="supported_fact", text="A fabricated source proves the lead.", source_ids=["source-fabricated"])],
            })])

    app.dependency_overrides[get_ai_client] = lambda: FabricatedSourceAI()
    response = client.post("/api/sidequests/surprise")

    assert response.status_code == 422
    assert "not supplied" in response.json()["detail"]


def test_detective_mode_suppresses_a_paraphrased_investigation(client: TestClient) -> None:
    thought_id = create_item(client, kind="thought", title="What if maps remembered where we hesitated?").json()["id"]
    client.post(f"/api/items/{thought_id}/concepts", json={"name": "Hesitation as spatial data", "category": "mechanism"})

    class ParaphraseAI(FakeAI):
        run = 0
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            self.run += 1
            item = contexts[0]["item_a"]
            if self.run == 1:
                clue = "A map records moments of hesitation as spatial traces of unrealized choices."
                lead = "Counterfactual cartography preserves routes that were considered but never taken."
                connection = "Hesitation traces can preserve unrealized choices and routes that were never taken."
            else:
                clue = "Moments of hesitation become spatial traces on a map of unrealized choices."
                lead = "Counterfactual cartography records considered routes that a traveler never took."
                connection = "Spatial hesitation traces preserve choices and considered routes that remain unrealized."
            primary = ConnectionCandidate(
                item_a_id=item["id"], item_b_id=item["id"], origin_item_ids=[item["id"]], title="The roads almost taken",
                clue=clue, clue_item_id=item["id"], clue_concept_names=["Hesitation as spatial data"], lead=lead,
                lead_kind="unfamiliar_subject", reasoning_steps=["Hesitation marks a still-open choice between routes.", "Recording that mark preserves a route that remained possible but unrealized."],
                claims=[SidequestClaim(claim_type="interpretation", text="A hesitation trace can represent an unrealized choice."), SidequestClaim(claim_type="speculative_hypothesis", text="Counterfactual maps might preserve routes not taken.")],
                connection=connection, bridge="Spatial records of unrealized choices and considered routes", why_interesting="It maps possibility rather than only movement.",
                where_it_breaks="A pause may reflect noise or distraction rather than a meaningful decision between possible routes.", new_question="Which unrealized routes deserve to remain visible?",
            )
            alternate = primary.model_copy(update={
                "title": "The pause as a navigational signal",
                "lead": "Wayfinding systems that expose decision friction",
                "lead_category": "practice",
                "connection": "Hesitation could be treated as feedback about where a route description creates uncertainty.",
                "bridge": "Decision friction revealing where navigation instructions fail",
                "new_question": "Which intersection designs repeatedly make first-time travelers pause before choosing a route?",
            })
            return CandidateBatch(candidates=[primary, alternate])

    ai = ParaphraseAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    assert client.post("/api/sidequests/surprise").status_code == 201
    repeated = client.post("/api/sidequests/surprise")
    assert repeated.status_code == 422
    assert "already been explored" in repeated.json()["detail"]


def test_sidequest_insufficient_data_and_rejected_candidate(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    assert client.post("/api/sidequests/surprise").status_code == 422
    first = create_item(client).json()["id"]
    second = create_item(client, kind="question", title="Why do flocks turn together?").json()["id"]
    for item_id in (first, second):
        client.post(f"/api/items/{item_id}/concepts", json={"name": "Coordination", "category": "mechanism"})
    fake = FakeAI()
    fake.reject = True
    app.dependency_overrides[get_ai_client] = lambda: fake
    with caplog.at_level("INFO", logger="app.main"):
        rejected = client.post("/api/sidequests/surprise")
    assert rejected.status_code == 422
    assert "No compelling case yet" in rejected.json()["detail"]
    assert "Too superficial" in rejected.json()["detail"]
    assert "left the investigation open rather than saving a weak connection" in rejected.json()["detail"]
    assert '"candidates_reviewed": 2' in caplog.text
    assert '"accepted": false' in caplog.text
    assert '"rejection_reason": "Too superficial"' in caplog.text
    assert '"outcome": "critic_rejected"' in caplog.text


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

    def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]], _curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
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
    def research(self, item: dict[str, object]) -> ResearchPacket:
        details = " ".join(detail for _kind, _label, detail in WorkFirstDiscoveryAI.anchors_by_title.get(str(item["title"]), []))
        source = RetrievedSource(
            url="https://example.com/publisher", title=f"{item['title']} publisher description",
            excerpt=details or "The source describes a specific work situation, its participants, constraints, and consequences.",
        )
        return ResearchPacket(synthesis="The publisher identifies the exact work and describes its concrete situations.", sources=[source, source])

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
        def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]], _curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
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
        def research(self, item: dict[str, object]) -> ResearchPacket:
            details = " ".join(
                f"A specific sequence {index} changes how characters coordinate under a clearly described constraint. "
                f"The retrieved work account describes sequence {index} and its concrete consequences for later choices."
                for index in range(1, 7)
            )
            source = RetrievedSource(
                url="https://example.com/publisher",
                title=f"{item['title']} publisher description",
                excerpt=details,
            )
            return ResearchPacket(
                synthesis="The publisher identifies the work and describes six concrete sequences.",
                sources=[source, source],
            )

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
        def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]], curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
            batch = super().identify_research_problems(understanding, work_sources, curiosity_profile)
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


def test_mixed_anchor_batch_keeps_valid_and_reports_discard_categories(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    received_profiles: list[dict[str, object]] = []

    class MixedAnchorAI(WorkFirstDiscoveryAI):
        def identify_research_problems(self, understanding: WorkUnderstanding, work_sources: list[dict[str, str]], curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
            received_profiles.append(curiosity_profile or {})
            valid = super().identify_research_problems(understanding, work_sources, curiosity_profile).problems[0]
            invalid_source = valid.model_copy(update={"problem_id": "bad-source", "evidence_source_ids": ["missing-source"]})
            duplicate = valid.model_copy(update={"concise_label": "A duplicate identifier must not replace the first anchor"})
            spoiler = valid.model_copy(update={"problem_id": "late-spoiler", "spoiler_scope": "full"})
            return ResearchProblemBatch(problems=[valid, invalid_source, duplicate, spoiler])

    item_id = create_item(
        client, title="The Hunger Games", progress_status="in_progress",
        progress_current=1, progress_total=3, progress_unit="book", spoiler_policy="through_progress",
    ).json()["id"]
    manual = client.post(f"/api/items/{item_id}/concepts", json={"name": "Communication across senses", "category": "personal"}).json()
    client.patch(f"/api/concepts/{manual['id']}/interest", json={"interested": True, "note": "I want to follow this."})
    app.dependency_overrides[get_ai_client] = lambda: MixedAnchorAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()

    with caplog.at_level("WARNING", logger="app.main"):
        response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    discoveries = [concept for concept in response.json()["concepts"] if concept["concept_classification"] == "grounded_observation"]
    assert [concept["name"] for concept in discoveries] == ["Spectacle as political control"]
    assert "kept=1 discarded=3" in caplog.text
    assert "'duplicate_id': 1" in caplog.text
    assert "'quality': 1" in caplog.text
    assert "'spoiler_boundary': 1" in caplog.text
    assert "A duplicate identifier must not replace" not in caplog.text
    assert "Communication across senses" in received_profiles[0]["authored_lenses"]


def test_exploration_invitation_is_idempotent_reversible_and_survives_exact_rerun(client: TestClient) -> None:
    item_id = create_item(client, title="The Hunger Games").json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: WorkFirstDiscoveryAI()
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()
    first = client.post(f"/api/items/{item_id}/enrich").json()
    concept = next(entry for entry in first["concepts"] if entry["name"] == "Spectacle as political control")

    saved = client.post(f"/api/concepts/{concept['id']}/exploration")
    repeated = client.post(f"/api/concepts/{concept['id']}/exploration")
    assert saved.status_code == 200
    assert repeated.json()["id"] == saved.json()["id"]
    assert saved.json()["question"] == concept["curiosity_hook"]
    assert saved.json()["status"] == "queued"

    dismissed = client.patch(f"/api/exploration-invitations/{saved.json()['id']}", json={"status": "dismissed", "note": "Not right now."})
    assert dismissed.status_code == 200
    assert dismissed.json()["status"] == "dismissed"
    assert dismissed.json()["user_note"] == "Not right now."
    assert client.post(f"/api/concepts/{concept['id']}/exploration").json()["status"] == "queued"

    rerun = client.post(f"/api/items/{item_id}/enrich").json()
    current = next(entry for entry in rerun["concepts"] if entry["name"] == concept["name"])
    persisted = client.get("/api/exploration-invitations", params={"item_id": item_id, "status": "queued"}).json()
    assert len(persisted) == 1
    assert persisted[0]["source_concept_id"] == concept["id"]
    assert persisted[0]["active_concept_id"] == current["id"]
    assert persisted[0]["available"] is True


def test_manual_concept_without_question_cannot_be_saved_as_exploration(client: TestClient) -> None:
    item_id = create_item(client).json()["id"]
    concept = client.post(f"/api/items/{item_id}/concepts", json={"name": "My lens", "category": "personal"}).json()

    response = client.post(f"/api/concepts/{concept['id']}/exploration")

    assert response.status_code == 422
    assert "specific exploration invitation" in response.json()["detail"]


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
    class ChangedObservationResearch(WorkOnlyResearch):
        def research(self, item: dict[str, object]) -> ResearchPacket:
            details = " ".join(detail for _kind, _label, detail in ai.anchors_by_title[str(item["title"])])
            source = RetrievedSource(
                url="https://example.com/publisher",
                title=f"{item['title']} publisher description",
                excerpt=details,
            )
            return ResearchPacket(
                synthesis="The publisher identifies the exact work and describes its concrete situations.",
                sources=[source, source],
            )

    app.dependency_overrides[get_research_provider] = lambda: ChangedObservationResearch()
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
        invitation_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(exploration_invitations)").fetchall()}
        sidequest_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(sidequests)").fetchall()}
    assert rows == [(1, "tv_series", "Castle"), (2, "movie", "An Ambiguous Title"), (3, "tv_series", "Game of Thrones")]
    assert preserved == (1, 1, 1, "needs_confirmation", 0, 0)
    assert p0_defaults == ("unknown", "unknown", "installment", "avoid")
    assert {"work_scope", "parent_work_title", "canonical_identity_id", "progress_status", "progress_current", "progress_total", "progress_unit", "spoiler_policy"} <= p0_columns
    assert {"item_id", "source_concept_id", "source_feedback_key", "question", "status", "user_note"} <= invitation_columns
    assert {"origin_item_ids_json", "discovery_kind", "invitation", "dedupe_key", "detective_version", "clue_json", "lead_json", "reasoning_json", "claims_json"} <= sidequest_columns

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


def test_interest_prioritizes_sidequest_clues_without_changing_verification(client: TestClient) -> None:
    first = create_item(client, title="First").json()["id"]
    second = create_item(client, title="Second").json()["id"]
    third = create_item(client, title="Third").json()["id"]
    concept_ids = []
    for item_id, name in [(first, "One mechanism"), (second, "Two mechanism"), (third, "Three mechanism")]:
        concept_ids.append(client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"}).json()["id"])
    for concept_id in concept_ids[:2]:
        assert client.patch(f"/api/concepts/{concept_id}/interest", json={"interested": True}).status_code == 200

    class CapturingAI(FakeAI):
        first_clue_id: int | None = None
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            self.first_clue_id = contexts[0]["item_a"]["id"]  # type: ignore[index]
            return super().generate_candidates(contexts)

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    assert client.post("/api/sidequests/surprise").status_code == 201
    assert ai.first_clue_id in {first, second}
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
        clue_ids: list[int] = []

        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            self.clue_ids = [context["item_a"]["id"] for context in contexts]  # type: ignore[index]
            return super().generate_candidates(contexts)

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert ai.clue_ids[0] == item_ids[0]
    assert set(ai.clue_ids) == set(item_ids)
    assert client.get(f"/api/items/{item_ids[0]}/concepts").json()[0]["concept_verification"] is None
    assert created.json()["evidence_status"].startswith("Exploratory")


def test_saved_invitation_is_bounded_sidequest_context_not_evidence(client: TestClient) -> None:
    class CapturingAI(WorkFirstDiscoveryAI):
        contexts: list[dict[str, object]] = []

        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            self.contexts = contexts
            return super().generate_candidates(contexts)

    ai = CapturingAI()
    app.dependency_overrides[get_ai_client] = lambda: ai
    app.dependency_overrides[get_research_provider] = lambda: WorkOnlyResearch()
    item_ids = [create_item(client, title=title, kind="tv_series" if title == "Castle" else "book").json()["id"] for title in ("Castle", "The Hunger Games", "A Psalm for the Wild-Built")]
    researched = [client.post(f"/api/items/{item_id}/enrich").json() for item_id in item_ids]
    seed = next(concept for concept in researched[0]["concepts"] if concept["curiosity_hook"])
    assert client.post(f"/api/concepts/{seed['id']}/exploration").status_code == 200

    created = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert ai.contexts[0]["item_a"]["id"] == item_ids[0]  # type: ignore[index]
    concepts = ai.contexts[0]["item_a"]["concepts"]  # type: ignore[index]
    saved_context = next(concept["saved_exploration"] for concept in concepts if concept["name"] == seed["name"])
    assert saved_context["question"] == seed["curiosity_hook"]
    assert created.json()["evidence"] == []
    assert created.json()["evidence_status"].startswith("Exploratory")


def test_sidequest_rejects_repeated_bridge_and_forces_exploratory_status(client: TestClient) -> None:
    first = create_item(client, title="First").json()["id"]
    second = create_item(client, title="Second").json()["id"]
    for item_id, name in [(first, "Distributed coordination"), (second, "Collective sensing")]:
        client.post(f"/api/items/{item_id}/concepts", json={"name": name, "category": "mechanism"})

    class OverclaimingAI(FakeAI):
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            stable_contexts = sorted(contexts, key=lambda context: int(context["item_a"]["id"]))  # type: ignore[index]
            return super().generate_candidates(stable_contexts)

        def critique_candidates(self, candidates: list[dict[str, object]], dossiers: list[dict[str, object]]) -> CritiquedConnection:
            result = super().critique_candidates(candidates, dossiers)
            return result.model_copy(update={"evidence_status": "Verified scientific relationship."})

    app.dependency_overrides[get_ai_client] = lambda: OverclaimingAI()
    created = client.post("/api/sidequests/surprise")
    repeated = client.post("/api/sidequests/surprise")

    assert created.status_code == 201
    assert created.json()["evidence_status"].startswith("Exploratory")
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


def test_generic_but_grounded_observation_fails_discovery_quality() -> None:
    source = {"work-1": "The protagonist uses humor in serious situations and jokes during a crisis to cope with stress."}
    problem = ResearchProblem(
        problem_id="generic-humor", concise_label="Humor in Crisis", anchor_kind="tone_or_humor",
        concrete_description="The protagonist uses humor in serious situations and jokes during a crisis.",
        work_situation="The protagonist uses humor during danger to cope with stress.",
        mechanism="Humor appears during dangerous events and helps the protagonist cope with stress.",
        interesting_reason="The humor makes difficult moments more approachable and gives the audience emotional relief.",
        distinctiveness="The story combines serious danger with humor instead of maintaining a consistently somber tone.",
        intellectual_question="Why does humor appear during difficult moments?",
        curiosity_invitation="Why do readers respond to humor during difficult moments?",
        relevant_domains=["narrative studies"], evidence_source_ids=["work-1"], research_queries=[],
        external_research_warranted=False,
        discovery_quality={"specificity": "high", "intellectual_substance": "medium", "surprise_potential": "low", "rationale": "Accurate but broad."},
    )

    assessment = assess_research_problem(problem, source)
    assert assessment["specificity"] == "high"  # factual grounding alone is not enough
    assert assessment["insight"] == "low"
    assert assessment["curiosity_potential"] == "low"
    assert "does not yet turn" in (research_problem_rejection(problem, {"work-1"}, "Example", source) or "")


def test_weak_anchor_is_dropped_without_failing_enrichment(client: TestClient) -> None:
    class GenericAnchorAI(FakeAI):
        def identify_research_problems(self, _understanding: WorkUnderstanding, work_sources: list[dict[str, str]], _curiosity_profile: dict[str, object] | None = None) -> ResearchProblemBatch:
            return ResearchProblemBatch(problems=[ResearchProblem(
                problem_id="generic-humor", concise_label="Humor in Crisis", anchor_kind="tone_or_humor",
                concrete_description="The protagonist uses humor in serious situations and jokes during a crisis.",
                work_situation="The protagonist uses humor during danger to cope with stress.",
                mechanism="Humor appears during dangerous events and helps the protagonist cope with stress.",
                interesting_reason="The humor makes difficult moments more approachable and gives the audience emotional relief.",
                distinctiveness="The story combines serious danger with humor instead of maintaining a consistently somber tone.",
                intellectual_question="Why does humor appear during difficult moments?",
                curiosity_invitation="Why do readers respond to humor during difficult moments?",
                relevant_domains=["narrative studies"], evidence_source_ids=[work_sources[0]["source_id"]], research_queries=[],
                external_research_warranted=False,
                discovery_quality={"specificity": "high", "intellectual_substance": "medium", "surprise_potential": "low", "rationale": "Accurate but broad."},
            )])

    class GenericWorkResearch(FakeResearch):
        def research(self, _item: dict[str, object]) -> ResearchPacket:
            return ResearchPacket(synthesis="A generic but sourced observation.", sources=[RetrievedSource(
                url="https://example.com/generic-humor", title="Publisher description",
                excerpt="The protagonist uses humor in serious situations and jokes during a crisis to cope with stress.",
            )])

    item_id = create_item(client).json()["id"]
    app.dependency_overrides[get_ai_client] = lambda: GenericAnchorAI()
    app.dependency_overrides[get_research_provider] = lambda: GenericWorkResearch()

    response = client.post(f"/api/items/{item_id}/enrich")

    assert response.status_code == 200
    assert response.json()["status"] == "researched"
    assert response.json()["concepts"] == []
    assert len(response.json()["sources"]) == 1


def test_distinctive_nontechnical_insight_passes_anchor_quality() -> None:
    source = {"work-1": "Mara folds her late mother's red scarf into a paper bird and leaves it on an empty chair before each performance."}
    problem = ResearchProblem(
        problem_id="ritual-grief", concise_label="Grief folded into readiness", anchor_kind="motif",
        concrete_description="Mara folds her late mother's red scarf into a paper bird before each performance.",
        work_situation="The paper bird remains on an empty chair while Mara performs for the audience.",
        mechanism="By repeating the private ritual before a public performance, the scarf turns inherited grief into a quiet form of readiness.",
        interesting_reason="A small tactile ritual lets private grief accompany public creativity without being resolved or explained away.",
        distinctiveness="The repeated transformation of the red scarf into a paper bird makes absence physically present without dialogue.",
        intellectual_question="How can a private ritual carry grief into public creative work?",
        curiosity_invitation="When can a repeated private ritual turn inherited grief into readiness for public performance?",
        relevant_domains=["ritual studies", "aesthetics"], evidence_source_ids=["work-1"], research_queries=[],
        external_research_warranted=False, appeal_dimensions=["beautiful", "emotionally_resonant"],
        discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "medium", "rationale": "A concrete emotional and formal mechanism."},
    )

    assessment = assess_research_problem(problem, source)
    assert assessment["specificity"] == "high"
    assert assessment["insight"] == "high"
    assert research_problem_rejection(problem, {"work-1"}, "Example", source) is None


def test_surprising_connection_between_form_and_coordination_scores_for_curiosity() -> None:
    source = {"work-1": "Rocky taps prime-number rhythms on the tunnel wall, and Grace answers with matching gestures until the two can coordinate a repair."}
    problem = ResearchProblem(
        problem_id="rhythm-coordination", concise_label="Arithmetic becomes choreography", anchor_kind="relationship",
        concrete_description="Rocky taps prime-number rhythms and Grace answers with matching gestures during the repair.",
        work_situation="Prime-number rhythms on the tunnel wall become matching gestures that let Rocky and Grace coordinate.",
        mechanism="By turning prime-number rhythms into matching gestures, the pair changes abstract arithmetic into embodied coordination.",
        interesting_reason="The interaction makes communication emerge as choreography before either participant shares a spoken language.",
        distinctiveness="Prime-number rhythms do not merely prove intelligence; they become a tempo for joint physical action.",
        intellectual_question="When can abstract patterns become tools for embodied cooperation?",
        curiosity_invitation="When can prime-number rhythms and matching gestures turn arithmetic into embodied coordination?",
        relevant_domains=["communication", "embodied cognition"], evidence_source_ids=["work-1"], research_queries=[],
        external_research_warranted=False, appeal_dimensions=["unexpected", "clever"],
        discovery_quality={"specificity": "high", "intellectual_substance": "high", "surprise_potential": "high", "rationale": "An unexpected bridge from mathematical form to joint action."},
    )

    assessment = assess_research_problem(problem, source)
    assert assessment["assessed"] is True
    assert assessment["specificity"] == "high"
    assert assessment["insight"] == "high"
    assert assessment["curiosity_potential"] == "high"
    assert research_problem_rejection(problem, {"work-1"}, "Example", source) is None


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


def generated_concept(
    concept_id: int, classification: str, name: str, hook: str, definition: str, mechanism: str,
) -> ItemConcept:
    return ItemConcept(
        id=concept_id, item_id=1, name=name, canonical_name=name, category="mechanism", origin="ai",
        definition=definition, explanation=definition, mechanism=mechanism, relationship_evidence=definition,
        concept_classification=classification, curiosity_hook=hook, discovery_quality_json='{"specificity":"high"}',
        concept_verification="verified" if classification == "established_concept" else "work_supported",
        relationship_verification="verified", interested=False, loved=False, surprised=False,
    )


def test_cross_category_dedupes_identical_curiosity_invitations_and_prefers_evidence() -> None:
    hook = "When can humor preserve clear coordination while a crew is facing mortal danger?"
    observation = generated_concept(1, "grounded_observation", "Humor in Crisis", hook, "Humor helps a stranded crew keep working through danger.", "Humor regulates stress while the crew coordinates repairs.")
    established = generated_concept(2, "established_concept", "Humor in Crisis", hook, "Humor can regulate stress and preserve group coordination during danger.", "Stress regulation protects communication and coordinated action.")

    visible = deduplicate_concept_candidates(
        [observation, established], {1: {"humor-anchor"}, 2: {"humor-anchor"}},
        {1: ["observation"], 2: ["concept", "relationship"]},
    )

    assert [concept.id for concept in visible] == [2]
    assert [observation.id, established.id] == [1, 2]  # persisted inputs remain untouched


def test_cross_category_dedupes_paraphrased_underlying_idea() -> None:
    observation = generated_concept(
        1, "grounded_observation", "Humor under pressure",
        "How does humor help a stranded crew coordinate clearly during mortal danger?",
        "Humor regulates stress and preserves coordination when a stranded crew faces lethal failures.",
        "Playful exchanges regulate stress and preserve coordinated action during lethal failures.",
    )
    interpretation = generated_concept(
        2, "grounded_interpretation", "Jokes as crisis coordination",
        "When can jokes help a stranded crew coordinate during life-threatening danger?",
        "Jokes regulate crew stress, preserving coordination through lethal equipment failure.",
        "Shared jokes regulate stress so the stranded crew can preserve coordination through equipment failure.",
    )

    assert concepts_are_semantic_duplicates(observation, interpretation, {"humor-anchor"}, {"humor-anchor"})
    assert len(deduplicate_concept_candidates([observation, interpretation], {1: {"humor-anchor"}, 2: {"humor-anchor"}})) == 1


def test_cross_category_keeps_distinct_observations_on_same_broad_topic() -> None:
    coordination = generated_concept(
        1, "grounded_observation", "Humor under pressure",
        "When can humor preserve coordination while a crew is facing mortal danger?",
        "Humor regulates stress and protects coordination during dangerous repair work.",
        "Stress regulation keeps communication usable during urgent joint action.",
    )
    dissent = generated_concept(
        2, "grounded_interpretation", "Humor as protected dissent",
        "When do jokes let subordinates challenge authority without direct confrontation?",
        "Humor can create social cover for criticism across a hierarchy.",
        "Ambiguous joking lowers the interpersonal cost of challenging a powerful authority figure.",
    )

    assert not concepts_are_semantic_duplicates(coordination, dissent, {"humor-anchor"}, {"humor-anchor"})
    assert len(deduplicate_concept_candidates([coordination, dissent], {1: {"humor-anchor"}, 2: {"humor-anchor"}})) == 2


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
        def generate_candidates(self, contexts: list[dict[str, object]]) -> CandidateBatch:
            a = contexts[0]["item_a"]["id"]  # type: ignore[index]
            b = contexts[0]["possible_library_leads"][0]["id"]  # type: ignore[index]
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
    assert "No compelling case yet" in response.json()["detail"]
    assert "broad shared topic" in response.json()["detail"]
