# Asterisk ✦

**There's more to the story.**

A personal Curiosity Library for quickly capturing books, movies, podcasts, articles, concepts, questions, and random thoughts. AI-assisted concept exploration and experimental **Sidequests** surface unexpected bridges across your interests.

## Requirements
- Python 3.10+
- Node.js 20.19+ (or 22.12+)

## One-time setup

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../frontend
npm install
```

## Start Asterisk

From the repository root, start the backend and frontend together:

```bash
./scripts/dev.sh
```

Open http://127.0.0.1:5173. The API is available at http://127.0.0.1:8000 and its interactive documentation is at http://127.0.0.1:8000/docs.

To run the services separately, use two terminals:

```bash
cd backend
.venv/bin/uvicorn app.main:app --reload
```

```bash
cd frontend
npm run dev
```

The frontend proxies `/api` requests to the backend. Data persists at `backend/data/asterisk.db`, using an absolute path derived from the application location rather than the terminal's working directory. Asterisk creates the data directory automatically, and `*.db` files are excluded from git. The library works without a cloud service; AI features require an OpenRouter API key.

If an earlier installation has `backend/curiosity.db`, Asterisk copies it to the new location on first startup when `backend/data/asterisk.db` does not yet exist. The original file is left untouched.

## API

- `GET /api/health`
- `GET /api/items?q=&kind=` — search/filter
- `POST /api/items` — add
- `PATCH /api/items/{id}` — edit
- `DELETE /api/items/{id}` — delete
- `GET /api/items/{id}/concepts` — list concepts
- `POST /api/items/{id}/concepts` — add a user-authored concept
- `POST /api/items/{id}/concepts/extract` — explicitly request AI extraction
- `GET /api/items/{id}/enrichment` — inspect research identity, sources, and evidence-linked concepts
- `POST /api/items/{id}/enrich` — explicitly research and enrich one item
- `DELETE /api/concepts/{id}` — remove a concept
- `GET /api/sidequests` — revisit saved Sidequests
- `POST /api/sidequests/surprise` — generate and save one Sidequest

Item kinds: `book`, `movie`, `podcast`, `article`, `concept`, `question`, and `thought` (shown as “Random thought” in the interface). Each item has a title, optional personal note and source URL, and an automatic creation timestamp.

## Configure AI features

Asterisk uses OpenRouter's OpenAI-compatible API with Structured Outputs. Copy the example configuration and paste your API key into the local `.env` file:

```bash
cp .env.example .env
# Edit .env and replace your_api_key_here.
./scripts/dev.sh
```

`scripts/dev.sh` loads the root `.env` automatically. The file is ignored by Git; never paste the key into `.env.example`, frontend code, or another tracked file.

`OPENROUTER_API_KEY` is required. `OPENROUTER_MODEL` defaults to `openai/gpt-4o-mini`, `OPENROUTER_TIMEOUT_SECONDS` defaults to `30`, and `OPENROUTER_SEARCH_ENGINE` defaults to `exa`. Choose a model whose OpenRouter page lists structured-output and tool-use support.

**Enrich with Research** is explicitly user-triggered and billable. It uses two bounded OpenRouter web-search stages—work understanding and problem-oriented concept research—plus structured calls for understanding, research-problem identification, evidence-grounded concept extraction, and relationship validation. Search-tool charges and ordinary model token charges apply. If either retrieval stage produces no inspectable sources, Asterisk preserves the previous enrichment and saves no fabricated research. Research permits full spoilers for items already in your library.

Extraction scans the full independent research corpus, not only passages that answer the first research problems. Accepted results distinguish **From the work** (`work_grounded`) from **Found while exploring** (`research_led`). Research-led concepts need independent definition and mechanism evidence plus a traceable research path, but may honestly report that no direct relationship to the saved work was established.

When a promising candidate fails only because its evidence cannot be verified, Asterisk first tries to rebind an exact excerpt already present elsewhere in the corpus, then performs one bounded semantic review of that same corpus. It may then make one targeted retrieval and evidence-repair attempt for a strictly limited number of candidates. Every proposed passage is still checked verbatim by the backend. `EVIDENCE_RECOVERY_MAX_CANDIDATES` (default `2`), `EVIDENCE_RECOVERY_MAX_SEARCH_RESULTS` (default `4`), and the estimated-spend settings in `.env.example` bound this extra work. Recovery never lowers the quality threshold, never retries automatically, and records the initial failure, recovery action, and final outcome for inspection. The spend limit is an internal estimate rather than a provider-enforced billing cap; actual search and token prices remain provider-dependent.

Concept extraction is always user-triggered. Quick extraction uses only the saved entry and is labeled accordingly. Research enrichment first maps concrete situations, phenomena, dilemmas, systems, and causal relationships in the work. It turns those observations into descriptive research problems without naming concepts, searches for explanations, and extracts either established terms or specific descriptive ideas grounded in the retrieved evidence. Descriptive labels may be original; their definitions, mechanisms, verbatim excerpts, opaque source references, and work relationships still receive strict independent checks. Only then does Asterisk validate concept truth separately from the concept’s explicit, illustrated, or interpretive relationship to the work. Insufficient evidence and rejected candidates remain inspectable with stage-specific reasons. Accepted concepts preserve label kind, independent definitions, domains, mechanisms, optional grounded curiosity hooks, both verification statuses, and supporting sources. Re-enrichment replaces prior AI research while preserving manually added concepts and original notes. Ambiguous named works are marked for confirmation; concepts, questions, and thoughts use subject research without book-like identity confirmation.

Rejected concept candidates can be reviewed in the Research & Concepts drawer. **Keep anyway** records interest without changing the automated rejection and creates a non-canonical discovery labeled **User-kept · Unverified** or **User-kept · Partially supported**. **Dismiss** and **Undo decision** are reversible and retain a timestamped review history. Re-evaluation checks saved and newly available evidence deterministically; it may update the human-facing support label, but never changes the original automated outcome. User-kept concepts are available to exploratory features, while their sources are deliberately withheld as claim-supporting evidence and they remain outside verified-only workflows.

Sidequests require concepts attached to at least two library items. In the Sidequests section, choose a curiosity and use **Add concepts** to generate suggestions or write your own; repeat this for any second item to unlock **Surprise Me**. The two items may have the same type—the critic still rejects connections that are too obvious or superficial. A “concept” here is a theme attached to any curiosity, not only an item saved with the Concept type. The experimental pipeline selects candidate pairs, generates possible connections, critiques them, rejects weak results, and persists the strongest accepted result. Every saved Sidequest is labeled exploratory and externally unverified.

Schema changes are additive at startup for the local SQLite database: research and lead-audit tables are created and new nullable/defaulted columns are added without deleting existing data. Canonical concepts are reusable records; item concepts hold the work-specific relationship and can link multiple library items to one canonical idea. Established terms can resolve through exact canonical names or verified aliases. Descriptive labels merge only on an exact normalized name, avoiding aggressive fuzzy consolidation of distinct ideas.

Run the offline evidence-first audit against the saved local research corpus without making network or model calls:

```bash
backend/.venv/bin/python backend/evals/evidence_first_saved_corpus.py
```

The audit includes explicit human-review fixtures for Rama and The Three-Body Problem, including work-grounded discovery, research-led corpus coverage, exact-evidence recovery, and candidates that must remain rejected. It does not claim that mocked or hand-reviewed extraction demonstrates live model quality.

## Tests

```bash
backend/.venv/bin/pytest

cd frontend
npm run build
```

Run the deterministic quality gate against the included weak examples:

```bash
backend/.venv/bin/python backend/evals/quality_eval.py backend/evals/weak_candidates.json
```

Candidates that pass this gate still require human review for surprise, conceptual depth, grounding, specificity, and curiosity potential; the gate is not a calibrated quality score.

## Roadmap
1. **Milestone 1**: capture, list, search, filter, edit, and delete
2. **Milestone 2 (current)**: user-triggered concept extraction and experimental Sidequests
3. Later: evidence gathering, feedback-driven ranking, and deeper discovery strategies

## Privacy
Library data and SQLite remain local. When you explicitly use Explore Concepts or Surprise Me, relevant titles, notes, URL strings, and concepts are sent to the configured AI provider. API keys remain in backend environment variables and are never sent to the browser or stored in Git.
