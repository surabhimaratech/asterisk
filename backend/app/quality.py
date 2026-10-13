import re
from urllib.parse import urlparse

from .ai import ConnectionCandidate, ValidatedConcept


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


GENERIC_CONCEPT_PATTERNS = (
    r"\b(?:themes?|narratives?|cultural impact|human response|human curiosity)\b",
    r"^(?:technology and ethics|fear in|cognitive limits and the unknown|navigating the unknown)",
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


def concept_quality_rejection(result: ValidatedConcept, work_urls: set[str], concept_urls: set[str], concept_source_text: dict[str, str] | None = None) -> str | None:
    name = result.canonical_name.strip().lower()
    if any(re.search(pattern, name) for pattern in GENERIC_CONCEPT_PATTERNS):
        return "Rejected as a broad theme, narrative framing, character description, or generic topic."
    if re.match(r"^(?:explores?|exploration|analys(?:is|es)|investigat(?:es|ing)|focuses|examines|highlights)\b", result.definition.strip(), re.I):
        return "The proposed definition describes what a work discusses instead of defining an independent concept."
    if not set(result.concept_source_urls) & concept_urls:
        return "No independent concept-research source supports the concept’s validity."
    if concept_source_text is not None:
        usable_urls = {
            url for url in result.concept_source_urls
            if url in concept_urls and _usable_concept_source(url, concept_source_text.get(url, ""))
        }
        if not usable_urls:
            return "The cited concept sources are inaccessible, low-value theme summaries, or lack substantive retrieved content."
        if len(name.split()) > 6:
            return "The proposed canonical name is too compound or work-specific to function as an established concept."
        normalized_name = re.sub(r"[^a-z0-9]+", " ", name).strip()
        cited_text = " ".join(concept_source_text.get(url, "") for url in usable_urls)
        normalized_text = re.sub(r"[^a-z0-9]+", " ", cited_text.lower())
        if normalized_name not in normalized_text:
            return "The proposed canonical name does not appear in the cited independent concept research."
    if not set(result.relationship_source_urls) & work_urls:
        return "No work-specific source supports the proposed relationship."
    if result.relationship_type == "interpretive" and result.relationship_verification == "verified":
        return "An interpretive relationship cannot be labeled externally verified."
    if len(result.mechanism.strip()) < 45:
        return "The lead does not explain a sufficiently specific mechanism or central problem."
    return None
