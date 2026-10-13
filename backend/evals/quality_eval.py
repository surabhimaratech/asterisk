"""Repeatable Sidequest candidate evaluation.

Usage: backend/.venv/bin/python backend/evals/quality_eval.py candidates.json
This deterministic gate is not a claim of calibrated discovery quality. Human review remains required.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import ConnectionCandidate
from app.quality import deterministic_rejection_reason


def main(path: str) -> int:
    candidates = json.loads(Path(path).read_text())
    for raw in candidates:
        candidate = ConnectionCandidate.model_validate(raw)
        reason = deterministic_rejection_reason(candidate)
        print(f"{'REJECT' if reason else 'REVIEW'}\t{candidate.title}\t{reason or 'passed deterministic checks; requires human rubric review'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
