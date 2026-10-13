import re

from .ai import ConnectionCandidate


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
