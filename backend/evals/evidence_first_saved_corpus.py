"""Offline evidence-first proof of concept over Asterisk's saved SQLite corpus.

This script makes no network or model calls. Its small review scenarios are inspectable
human-authored fixtures for testing orchestration and evidence sufficiency, not production
rules or proof of live-model quality.

Usage:
    backend/.venv/bin/python backend/evals/evidence_first_saved_corpus.py
    backend/.venv/bin/python backend/evals/evidence_first_saved_corpus.py path/to/asterisk.db
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "asterisk.db"

FIXTURE_SCENARIOS = [
    {
        "title": "Rendezvous with Rama",
        "problems": [
            "How can investigators infer the purpose of an unfamiliar artificial structure from limited observations?",
            "What limits attempts to infer the intentions of radically unfamiliar creators?",
            "How should explorers act when important properties of an artifact remain unknown?",
        ],
        "candidates": [
            {
                "name": "Worldmaking through linguistic categories",
                "discovery_origin": "work_grounded",
                "research_path": "Work-grounded questions about interpreting Rama's radically unfamiliar makers led to independent philosophical research on how unfamiliar languages may organize reality differently.",
                "label_kind": "descriptive_label",
                "definition": "Linguistic systems can present reality differently when their expressions divide meaning into unfamiliar kinds.",
                "mechanism": "Semantic categories shape which entities and processes a language makes readily expressible, so a radically unfamiliar language may encode a different apparent ontology.",
                "concept_source_ids": [58],
                "definition_evidence": "Languages present the world as being in different ways depending on what kinds of meanings their expressions have.",
                "relationship_source_ids": [47],
                "relationship_type": "interpretive",
                "relationship": "Rama leaves the crew with physical evidence but no conversation with its makers, leaving open whether human conceptual categories fit what they encountered.",
                "curiosity_hook": "Could a language built without nouns make its speakers experience a world without stable things?",
                "status": "accept",
                "reason": "The descriptive label captures a specific source-supported mechanism, while a separate work excerpt supports only a clearly marked interpretive relationship.",
            },
            {
                "name": "Messaging extraterrestrial intelligence",
                "label_kind": "established_term",
                "aliases": ["METI"],
                "discovery_origin": "research_led",
                "research_path": "The independent corpus broadened from first-contact uncertainty into ethical and political research about deliberately sending messages to extraterrestrial civilizations.",
                "definition": "The deliberate transmission of communications intended for possible extraterrestrial intelligence.",
                "mechanism": "A sender intentionally encodes and transmits a message beyond Earth despite uncertainty about recipients, authority, and possible consequences.",
                "concept_source_ids": [55],
                "definition_evidence": "The decision on whether or not to send a message to extraterrestrial intelligence should be made by an appropriate international body, broadly representative of Humankind.",
                "relationship_source_ids": [],
                "relationship_type": "unsupported",
                "relationship": "No direct claim that the novel discusses METI is required; this is a research-led branch from the independent corpus.",
                "curiosity_hook": "Who can legitimately consent when one group sends a message on behalf of an entire planet?",
                "status": "accept",
                "reason": "The saved independent corpus establishes a specific transferable practice and governance problem; the provenance explicitly avoids claiming a direct work relationship.",
            },
            {
                "name": "Human Understanding and Alien Complexity",
                "discovery_origin": "work_grounded",
                "research_path": "A broad search about human responses to unfamiliar artifacts produced this proposed label, but not enough specific independent support.",
                "label_kind": "descriptive_label",
                "definition": "A broad description of difficulty understanding unfamiliar things.",
                "mechanism": "The proposal does not identify a concrete process beyond general difficulty and unfamiliarity.",
                "concept_source_ids": [55],
                "definition_evidence": "awe is indeed associated with epistemic dispositions and facilitates the exploration of uncertainty",
                "relationship_source_ids": [45],
                "relationship_type": "interpretive",
                "relationship": "Interpretive: the artifact's scale is difficult for humans to comprehend.",
                "curiosity_hook": "What can we learn from alien complexity?",
                "status": "reject",
                "reason": "The label and hook are generic, and the saved independent excerpt does not establish this synthesized concept.",
            },
            {
                "name": "Technosignature",
                "discovery_origin": "work_grounded",
                "research_path": "The artificial-object research problem suggested technosignatures, but the bounded saved corpus did not retrieve a valid independent definition.",
                "label_kind": "established_term",
                "definition": "Observable evidence of technology beyond Earth.",
                "mechanism": "Observers test whether an astronomical signal or artifact is better explained by technology than by natural processes.",
                "concept_source_ids": [],
                "definition_evidence": "",
                "relationship_source_ids": [47],
                "relationship_type": "explicit",
                "relationship": "The detected cylinder is an artificial extraterrestrial object.",
                "curiosity_hook": "Which observations would separate a technosignature from an unusual natural object?",
                "status": "reject",
                "reason": "Inadequate retrieval: the saved Rama corpus contains no substantive source defining technosignatures.",
            },
        ],
    },
    {
        "title": "The Three Body Problem",
        "problems": [
            "Who has legitimate authority to send a message on behalf of humanity?",
            "How should a decision be made when extraterrestrial recipients' intentions are unknowable?",
        ],
        "candidates": [
            {
                "name": "Messaging extraterrestrial intelligence",
                "discovery_origin": "work_grounded",
                "research_path": "A work-grounded question about who may contact extraterrestrial intelligence led to independent policy and ethics sources defining deliberate messaging.",
                "label_kind": "established_term",
                "aliases": ["METI"],
                "definition": "The deliberate transmission of signals intended for possible extraterrestrial intelligence.",
                "mechanism": "A sender encodes and transmits information beyond Earth despite uncertainty about who may receive it and what follows.",
                "concept_source_ids": [37, 38, 41],
                "definition_evidence": "The act of sending deliberate signals into space has generated debate within the scientific and ethical communities",
                "relationship_source_ids": [34],
                "relationship_type": "explicit",
                "relationship": "Explicit: Ye Wenjie contacts the Trisolarans and encourages their intervention.",
                "curiosity_hook": "Who can legitimately choose to speak for humanity when sending an irreversible message?",
                "status": "accept",
                "reason": "Independent sources establish the term and governance mechanism; a separate work source establishes deliberate contact.",
            },
            {
                "name": "Learning from failed predictions",
                "discovery_origin": "work_grounded",
                "research_path": "A question about unstable prediction led toward model revision, but the saved source did not contain the claimed supporting passage.",
                "label_kind": "descriptive_label",
                "definition": "Revising a model when its predictions fail.",
                "mechanism": "Prediction errors expose assumptions in a model and guide which assumptions should be reconsidered.",
                "concept_source_ids": [39],
                "definition_evidence": "Failed predictions force models to reveal the hidden assumptions that should be revised.",
                "relationship_source_ids": [34],
                "relationship_type": "interpretive",
                "relationship": "Interpretive: the novel presents conflict with an alien civilization.",
                "curiosity_hook": "How can repeated prediction failure improve a model of a chaotic system?",
                "status": "reject",
                "reason": "The proposed quote is not present in source 39, and the work relationship does not support the claimed mechanism.",
            },
        ],
    },
]


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def main(db_path: str | None = None) -> int:
    path = Path(db_path) if db_path else DEFAULT_DB
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    report = {
        "database": str(path),
        "network_calls": 0,
        "fixture_kind": "human-authored deterministic review scenarios",
        "coverage_checks": [
            "work-grounded candidate with separate concept and work evidence",
            "research-led candidate discovered elsewhere in the independent corpus",
            "candidate rejected despite a real but semantically unrelated source",
        ],
        "recovery_checks": {
            "exact_excerpt_rebinding": "Covered by the offline quality test using a verbatim passage attached to the wrong source ID.",
            "targeted_retrieval": "Covered by mocked route tests: one successful recovery and one bounded failure retained in the audit trail.",
            "quality_thresholds": "The same deterministic and structured validation runs after recovery; no threshold is changed.",
        },
        "works": [],
    }
    for scenario in FIXTURE_SCENARIOS:
        item = connection.execute("SELECT id, title FROM curiosity_items WHERE title = ?", (scenario["title"],)).fetchone()
        if item is None:
            report["works"].append({"title": scenario["title"], "error": "work not found"})
            continue
        sources = {
            row["id"]: row
            for row in connection.execute("SELECT id, title, source_type, excerpt, url FROM research_sources WHERE item_id = ?", (item["id"],))
        }
        candidates = []
        for candidate in scenario["candidates"]:
            concept_sources = [sources[source_id] for source_id in candidate["concept_source_ids"] if source_id in sources]
            relationship_sources = [sources[source_id] for source_id in candidate["relationship_source_ids"] if source_id in sources]
            evidence = normalized(candidate["definition_evidence"])
            evidence_found = bool(evidence) and any(evidence in normalized(f"{source['title']} {source['excerpt']}") for source in concept_sources)
            candidates.append({
                **candidate,
                "initial_validation_outcome": "candidate evidence passed initial inspection" if evidence_found else "candidate evidence failed initial inspection",
                "recovery_eligibility": "not needed" if evidence_found else "eligible only within configured candidate and estimated-cost limits",
                "final_outcome": candidate["status"],
                "concept_sources_found": [{"id": source["id"], "title": source["title"], "role": source["source_type"]} for source in concept_sources],
                "relationship_sources_found": [{"id": source["id"], "title": source["title"], "role": source["source_type"]} for source in relationship_sources],
                "definition_evidence_found": evidence_found,
            })
        saved_concepts = [dict(row) for row in connection.execute("SELECT name, label_kind, definition, mechanism, relationship_type, relationship_evidence, curiosity_hook FROM item_concepts WHERE item_id = ? AND origin = 'ai'", (item["id"],))]
        saved_rejections = [dict(row) for row in connection.execute("SELECT name, stage, rejection_reason FROM concept_lead_audits WHERE item_id = ? AND accepted = 0", (item["id"],))]
        report["works"].append({
            "title": item["title"],
            "research_problems": scenario["problems"],
            "fixture_candidates": candidates,
            "actual_saved_model_results": {"accepted_concepts": saved_concepts, "rejected_audits": saved_rejections},
        })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
