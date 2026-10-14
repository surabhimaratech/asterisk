import re
from urllib.parse import urlparse

from .ai import ConnectionCandidate, EvidenceConceptCandidate, EvidenceReference, ResearchProblem, ValidatedConcept


GENERIC_PATTERNS = (
    r"both (?:works|items|entries).*(?:explore|discuss|deal with)",
    r"both.*(?:science|technology|humanity|aliens|mysteries)",
    r"different (?:formats|perspectives|genres)",
    r"one is fictional.*other.*nonfiction",
)


def deterministic_rejection_reason(candidate: ConnectionCandidate) -> str | None:
    combined = f"{candidate.title} {candidate.connection} {candidate.bridge}".lower()
    if any(re.search(pattern, combined) for pattern in GENERIC_PATTERNS):
        return "The candidate relies on a broad shared topic or format comparison."
    if len(candidate.bridge.strip()) < 24:
        return "The candidate does not name a sufficiently specific conceptual bridge."
    if len(candidate.where_it_breaks.strip()) < 30:
        return "The candidate does not explain the limits of the comparison."
    return None


def research_problem_rejection(problem: ResearchProblem, work_source_ids: set[str], work_title: str) -> str | None:
    cited_ids = set(problem.evidence_source_ids)
    if not cited_ids or not cited_ids <= work_source_ids:
        return "The research problem cites evidence that is not present in the retrieved work corpus."
    if not problem.concise_label.strip() or len(problem.interesting_reason.strip()) < 35:
        return "The work anchor does not clearly state what makes the concrete detail interesting."
    if len(problem.distinctiveness.strip()) < 25:
        return "The work anchor does not explain why it is distinctive rather than a generic theme."
    mechanism = (problem.mechanism or problem.interesting_reason).strip()
    if len(mechanism) < 35:
        return "The work anchor does not explain a concrete mechanism, tension, rule, or emotional dynamic."
    invitation = (problem.curiosity_invitation or problem.intellectual_question).strip()
    if len(invitation) < 30:
        return "The curiosity invitation is too vague to open a meaningful direction."
    if re.search(r"^(?:how does this (?:influence|affect)|why is this important|what can we learn|how do relationships affect)", invitation, re.I):
        return "The curiosity invitation restates a generic theme instead of opening a specific exploration."
    if problem.external_research_warranted and not problem.research_queries:
        return "The work anchor requests external research without a descriptive search query."
    generic_anchor = re.sub(r"[^a-z0-9]+", " ", f"{problem.concise_label} {problem.concrete_description}".casefold())
    if any(phrase in generic_anchor for phrase in ("themes of", "human nature", "good versus evil", "societal change", "character development")):
        return "The proposed work anchor is a broad theme rather than a distinctive observation."
    normalized_title = re.sub(r"[^a-z0-9]+", " ", work_title.casefold()).strip()
    for query in problem.research_queries:
        normalized_query = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
        if normalized_title and normalized_title in normalized_query:
            return "The research query is anchored to the work title instead of describing the intellectual problem independently."
    return None


GENERIC_CONCEPT_PATTERNS = (
    r"\b(?:themes?|narratives?|cultural impact|human response|human curiosity)\b",
    r"^(?:technology and ethics|fear in|cognitive limits and the unknown|navigating the unknown)",
    r"^(?:human understanding and alien complexity|challenges of encountering aliens|mysteries of alien design|psychological responses to exploration)$",
    r"\b(?:mystery and ambiguity|characterization)\b",
)

UNUSABLE_SOURCE_PATTERNS = (
    r"client challenge",
    r"required part of this site couldn.t load",
    r"enable javascript",
    r"access denied",
    r"page not found",
)

LOW_VALUE_CONCEPT_HOSTS = {
    "gradesaver.com",
    "www.gradesaver.com",
    "litcharts.com",
    "www.litcharts.com",
    "supersummary.com",
    "www.supersummary.com",
}


def _usable_concept_source(url: str, text: str) -> bool:
    normalized = " ".join(text.split()).lower()
    if len(normalized) < 80 or any(re.search(pattern, normalized) for pattern in UNUSABLE_SOURCE_PATTERNS):
        return False
    return (urlparse(url).hostname or "").lower() not in LOW_VALUE_CONCEPT_HOSTS


def candidate_evidence_rejection(candidate: EvidenceConceptCandidate, concept_source_text: dict[str, str], concept_source_urls: dict[str, str] | None = None) -> str | None:
    cited_ids = {reference.source_id for reference in candidate.concept_evidence}
    if not cited_ids:
        return "No independent evidence is attached to the candidate."
    if not cited_ids <= set(concept_source_text):
        return "A candidate evidence reference does not match any retrieved independent source."
    for reference in candidate.concept_evidence:
        text = concept_source_text[reference.source_id]
        if not _usable_concept_source((concept_source_urls or {}).get(reference.source_id, reference.source_id), text):
            return "A cited source is inaccessible, low-value, or lacks substantive retrieved content."
        evidence = re.sub(r"[^a-z0-9]+", " ", reference.excerpt.casefold()).strip()
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
        if len(evidence.split()) < 8:
            return "A cited evidence passage is too short to substantiate the candidate."
        if reference.evidence_form == "quotation" and evidence not in normalized_text:
            return "A cited excerpt is not a substantive verbatim passage from its referenced source."
    if candidate.label_kind == "established_term":
        cited_text = re.sub(r"[^a-z0-9]+", " ", " ".join(concept_source_text[source_id] for source_id in cited_ids).casefold())
        terms = [candidate.canonical_name, *candidate.aliases]
        if not any((term := re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()) and term in cited_text for value in terms):
            return "Neither the established term nor an evidence-established alias appears in its cited research."
    return None


def rebind_exact_evidence(candidate: EvidenceConceptCandidate, concept_source_text: dict[str, str]) -> tuple[EvidenceConceptCandidate, list[dict[str, str]]]:
    """Repair only an exact excerpt attached to the wrong source ID; never invent evidence."""
    rebound: list[EvidenceReference] = []
    events: list[dict[str, str]] = []
    for reference in candidate.concept_evidence:
        if reference.evidence_form != "quotation":
            rebound.append(reference)
            continue
        evidence = re.sub(r"[^a-z0-9]+", " ", reference.excerpt.casefold()).strip()
        current = re.sub(r"[^a-z0-9]+", " ", concept_source_text.get(reference.source_id, "").casefold()).strip()
        if evidence and evidence in current:
            rebound.append(reference)
            continue
        matches = [source_id for source_id, text in concept_source_text.items() if evidence and evidence in re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()]
        if len(matches) == 1:
            rebound.append(EvidenceReference(source_id=matches[0], excerpt=reference.excerpt, evidence_form="quotation"))
            events.append({"from_source_id": reference.source_id, "to_source_id": matches[0], "outcome": "exact_excerpt_rebound"})
        else:
            rebound.append(reference)
    return candidate.model_copy(update={"concept_evidence": rebound}), events


def concept_quality_rejection(result: ValidatedConcept, work_source_ids: set[str], concept_source_text: dict[str, str], concept_source_urls: dict[str, str] | None = None, work_title: str = "") -> str | None:
    name = result.canonical_name.strip().lower()
    if any(re.search(pattern, name) for pattern in GENERIC_CONCEPT_PATTERNS) and len(result.mechanism.strip()) < 80:
        return "Rejected as a broad theme, narrative framing, character description, or generic topic."
    if result.concept_classification == "speculative_connection":
        return "This is a plausible speculative connection, not an automatically verified discovery. Review it and keep it if it is worth exploring."
    if result.concept_classification == "established_concept" and result.label_kind != "established_term":
        return "An established concept must use terminology independently established by its evidence."
    if result.concept_classification == "grounded_interpretation" and result.relationship_verification == "unverified":
        return "A grounded interpretation requires work evidence supporting the interpretation."
    if result.mechanism_defensibility == "unsupported":
        return "The proposed mechanism is not defensible from the supplied evidence."
    normalized_title = re.sub(r"[^a-z0-9]+", " ", work_title.casefold()).strip()
    if normalized_title and normalized_title in re.sub(r"[^a-z0-9]+", " ", name):
        return "The proposed label is tied to this work rather than transferable to other contexts."
    if re.match(r"^(?:explores?|exploration|analys(?:is|es)|investigat(?:es|ing)|focuses|examines|highlights)\b", result.definition.strip(), re.I):
        return "The proposed definition describes what a work discusses instead of defining an independent concept."
    cited_ids = {reference.source_id for reference in result.concept_evidence}
    if not cited_ids:
        return "No independent concept-research source supports the concept’s validity."
    if not cited_ids <= set(concept_source_text):
        return "A concept evidence reference does not match any retrieved independent source."
    usable_ids = {
        source_id for source_id in cited_ids
        if _usable_concept_source((concept_source_urls or {}).get(source_id, source_id), concept_source_text.get(source_id, ""))
    }
    if usable_ids != cited_ids:
        return "The cited concept sources are inaccessible, low-value theme summaries, or lack substantive retrieved content."
    if len(name.split()) > 8:
        return "The proposed label is too compound or work-specific to function as a transferable concept."
    cited_text = " ".join(concept_source_text[source_id] for source_id in usable_ids)
    normalized_text = re.sub(r"[^a-z0-9]+", " ", cited_text.lower())
    if result.label_kind == "established_term":
        supported_terms = [result.canonical_name, *result.aliases]
        normalized_terms = [re.sub(r"[^a-z0-9]+", " ", term.lower()).strip() for term in supported_terms if term.strip()]
        if not any(term and term in normalized_text for term in normalized_terms):
            return "Neither the canonical name nor an evidence-established alias appears in the cited independent research."
    for reference in result.concept_evidence:
        evidence = re.sub(r"[^a-z0-9]+", " ", reference.excerpt.lower()).strip()
        source_text = re.sub(r"[^a-z0-9]+", " ", concept_source_text[reference.source_id].lower()).strip()
        if len(evidence.split()) < 8:
            return "The definition evidence is too short to substantiate the concept."
        if reference.evidence_form == "quotation" and evidence not in source_text:
            return "The definition evidence is not a substantive verbatim excerpt from its referenced concept source."
    discovery_ids = set(result.discovery_source_ids)
    if not discovery_ids or not discovery_ids <= set(concept_source_text):
        return "The discovery path cites a source that is absent from the independent research corpus."
    if len(result.research_path.strip()) < 50:
        return "The discovery path does not substantively explain how research led to the concept."
    has_work_support = bool(set(result.relationship_source_ids) & work_source_ids)
    if result.discovery_origin == "work_grounded":
        if not has_work_support:
            return "No work-specific source supports the proposed relationship."
        if result.relationship_type == "unsupported":
            return "A work-grounded discovery cannot have an unsupported work relationship."
    elif result.relationship_type != "unsupported" and not has_work_support:
        return "A claimed direct work relationship lacks work-specific evidence."
    elif result.relationship_type == "unsupported" and result.relationship_verification != "unverified":
        return "A research-led discovery without a direct work relationship must remain unverified against the work."
    if result.relationship_type == "interpretive" and result.relationship_verification == "verified":
        return "An interpretive relationship cannot be labeled externally verified."
    if len(result.mechanism.strip()) < 45:
        return "The lead does not explain a sufficiently specific mechanism or central problem."
    hook = (result.curiosity_hook or "").strip()
    if hook:
        if len(hook) < 30 or not hook.endswith("?"):
            return "The curiosity hook must be a substantive, open question."
        if re.search(r"^(?:what can we learn|how does this relate|why is this interesting)", hook, re.I):
            return "The curiosity hook is generic rather than concept-specific."
        grounding = set(re.sub(r"[^a-z0-9]+", " ", f"{result.canonical_name} {result.definition} {result.mechanism} {result.relationship_evidence}".lower()).split())
        hook_terms = set(re.sub(r"[^a-z0-9]+", " ", hook.lower()).split()) - {"what", "when", "where", "which", "could", "would", "does", "this", "that", "from", "with", "into", "about", "have", "been", "were", "will", "your", "their"}
        if len(hook_terms & grounding) < 2:
            return "The curiosity hook introduces a premise not grounded in the concept or its work relationship."
    return None
