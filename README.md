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

**Enrich with Research** is explicitly user-triggered and billable. It uses two bounded OpenRouter web-search stages—work understanding and targeted concept research—plus structured calls for understanding, lead generation, and validation. Search-tool charges and ordinary model token charges apply. If either retrieval stage produces no inspectable sources, Asterisk preserves the previous enrichment and saves no fabricated research. Research permits full spoilers for items already in your library.

Concept extraction is always user-triggered. Quick extraction uses only the saved entry and is labeled accordingly. Research enrichment first maps concrete situations, phenomena, dilemmas, systems, and causal relationships in the work. It then proposes conceptual leads, researches those concepts independently, and validates concept truth separately from the concept’s explicit, illustrated, or interpretive relationship to the work. Broad or generic leads are retained only as inspectable rejections. Accepted concepts preserve definitions, domains, mechanisms, both verification statuses, and supporting sources. Re-enrichment replaces prior AI research while preserving manually added concepts and original notes. Ambiguous named works are marked for confirmation; concepts, questions, and thoughts use subject research without book-like identity confirmation.

Sidequests require concepts attached to at least two library items. In the Sidequests section, choose a curiosity and use **Add concepts** to generate suggestions or write your own; repeat this for any second item to unlock **Surprise Me**. The two items may have the same type—the critic still rejects connections that are too obvious or superficial. A “concept” here is a theme attached to any curiosity, not only an item saved with the Concept type. The experimental pipeline selects candidate pairs, generates possible connections, critiques them, rejects weak results, and persists the strongest accepted result. Every saved Sidequest is labeled exploratory and externally unverified.

Schema changes are additive at startup for the local SQLite database: research and lead-audit tables are created and new nullable/defaulted columns are added without deleting existing data. Canonical concepts are reusable records; item concepts hold the work-specific relationship and can link multiple library items to one canonical idea.

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
