# Asterisk ✦

**There's more to the story.**

A personal Curiosity Library for quickly capturing books, movies, TV series, podcasts, articles, concepts, questions, and random thoughts. AI-assisted concept exploration and experimental **Sidequests** surface unexpected bridges across your interests.

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
- `POST /api/items/{id}/identity/resolve` — resolve a work before concept research
- `POST /api/items/{id}/identity/confirm` — choose and persist a disambiguation candidate
- `DELETE /api/items/{id}/identity` — correct/reset a persisted identity
- `POST /api/items/{id}/enrich` — explicitly research and enrich one item
- `DELETE /api/concepts/{id}` — remove a concept
- `PATCH /api/concepts/{id}/interest` — mark or unmark a concept as interesting
- `PATCH /api/concepts/{id}/feedback` — record reversible Love or Surprise feedback
- `PATCH /api/rejected-findings/{id}/interest` — record interest independently of review status
- `PATCH /api/rejected-findings/{id}/feedback` — record Love or Surprise without changing evidence status
- `GET /api/sidequests` — revisit saved Sidequests
- `POST /api/sidequests/surprise` — generate and save one Sidequest

Item kinds: `book`, `movie`, `tv_series`, `podcast`, `article`, `concept`, `question`, and `thought` (shown as “Random thought” in the interface). Each item has a title, optional personal note, source URL, creator and release year, plus an automatic creation timestamp. For works, the lightweight “Identity, progress & spoilers” control can also distinguish an individual work, series, collection, installment, or adaptation; record partial progress; and set a spoiler boundary.

## Configure AI features

Asterisk uses OpenRouter's OpenAI-compatible API with Structured Outputs. Copy the example configuration and paste your API key into the local `.env` file:

```bash
cp .env.example .env
# Edit .env and replace your_api_key_here.
./scripts/dev.sh
```

`scripts/dev.sh` loads the root `.env` automatically. The file is ignored by Git; never paste the key into `.env.example`, frontend code, or another tracked file.

`OPENROUTER_API_KEY` is required. `OPENROUTER_MODEL` defaults to `openai/gpt-4o-mini`, `OPENROUTER_TIMEOUT_SECONDS` defaults to `30`, and `OPENROUTER_SEARCH_ENGINE` defaults to `exa`. Choose a model whose OpenRouter page lists structured-output and tool-use support.

**Enrich with Research** is explicitly user-triggered and billable. For books, movies, TV series, podcasts, and articles, Asterisk first resolves the exact work. An exact high-confidence title, media, and work-scope match is persisted; multiple plausible matches stop at a disambiguation screen. A series cannot silently resolve to its first installment. Confirmed title, creator, year, media type, scope, parent work, and reliable canonical identifier are reused by later searches. Correcting an identity retires stale AI findings while preserving notes, manual concepts, and audit history.

Research defaults to premise-only material when progress is unknown. A saved progress boundary is included in work research, extraction, and Sidequest context, and generated observations carry an explicit spoiler scope that the backend checks before persistence. Changing progress or spoiler preferences supersedes current AI research so it must be regenerated under the new boundary; it does not delete the confirmed identity or user-authored concepts. Source records separately identify evidence used for work identity, work details, and external concepts, while identity-source snapshots remain inspectable.

After identity resolution, enrichment first identifies several specific anchors inside the work: character tensions, worldbuilding rules, narrative techniques, recurring motifs, striking systems, or revealing situations. It deterministically selects a diverse set of roughly 3–5 grounded observations, then researches only the strongest configured subset. `DISCOVERY_MAX_ANCHORS` defaults to `5` and `DISCOVERY_MAX_RESEARCH_ANCHORS` defaults to `3`. This selection adds no model call, and grounded observations remain useful discoveries even when they do not lead to an external academic concept or Sidequest.

The selected observations support two discovery modes: **deepen fascination**, which follows something already compelling in the work, and **reveal an overlooked perspective**, which surfaces a less obvious lens. External research still uses bounded OpenRouter work research, evidence-grounded concept extraction, and relationship validation. Search-tool charges and ordinary model token charges apply. If external research fails, Asterisk does not invent a concept; the separately work-supported observations remain available. Research permits full spoilers only when the item is marked finished/re-reading/rewatching or the user explicitly allows them.

Extraction scans the full independent research corpus, not only passages that answer the first research problems. Accepted results distinguish **From the work** (`work_grounded`) from **Found while exploring** (`research_led`). Research-led concepts need independent definition and mechanism evidence plus a traceable research path, but may honestly report that no direct relationship to the saved work was established.

When a promising candidate fails only because its evidence cannot be verified, Asterisk first tries to rebind an exact excerpt already present elsewhere in the corpus, then performs one bounded semantic review of that same corpus. It may then make one targeted retrieval and evidence-repair attempt for a strictly limited number of candidates. Every proposed passage is still checked verbatim by the backend. `EVIDENCE_RECOVERY_MAX_CANDIDATES` (default `2`), `EVIDENCE_RECOVERY_MAX_SEARCH_RESULTS` (default `4`), and the estimated-spend settings in `.env.example` bound this extra work. Recovery never lowers the quality threshold, never retries automatically, and records the initial failure, recovery action, and final outcome for inspection. The spend limit is an internal estimate rather than a provider-enforced billing cap; actual search and token prices remain provider-dependent.

Concept extraction is always user-triggered. Quick extraction uses only the saved entry and is labeled accordingly. Research results distinguish **Established concept**, **Grounded interpretation**, and **Speculative connection**. A generated interpretive label does not need to appear verbatim when its underlying mechanism and relationship are supported. Actual quotations remain subject to exact source-text checks; clearly labeled cited paraphrases may use different wording but still require a retrieved, substantive source and structured validation. Speculative connections remain in review unless the user explicitly keeps them.

Rejected concept candidates can be reviewed in the Research & Concepts drawer. **Interested**, **Love**, and **Surprise** are independent from **Keep anyway**, **Dismiss**, and **Re-evaluate**: they record personal discovery value without asserting truth. Feedback is copied across reruns only when a stable fingerprint of the underlying finding matches; similar titles alone are not enough. Keep anyway creates a non-canonical discovery labeled **User-kept · Unverified** or **User-kept · Partially supported** without changing the automated rejection. Reversible review history and evidence outcomes remain intact. User-kept concepts are available to exploratory features, while their sources are deliberately withheld as claim-supporting evidence and they remain outside verified-only workflows.

Sidequests require discoveries attached to at least two library items. Interest, Love, and Surprise contribute separate bounded ranking signals, while mechanism depth, grounded discoveries, domain/type diversity, and repetition penalties keep one clicked topic from dominating. Ranking never upgrades verification, and it retains an unfeedbacked exploration slot when possible. The two items may have the same type, and the critic still rejects connections that are too obvious or superficial. Every saved Sidequest remains exploratory and externally unverified.

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
npm test
npm run build
```

Run the deterministic quality gate against the included weak examples:

```bash
backend/.venv/bin/python backend/evals/quality_eval.py backend/evals/weak_candidates.json
```

Candidates that pass this gate still require human review for surprise, conceptual depth, grounding, specificity, and curiosity potential; the gate is not a calibrated quality score.

## Roadmap
1. **Milestone 1**: capture, list, search, filter, edit, and delete
2. **Milestone 2**: user-triggered concept extraction and experimental Sidequests
3. **Milestone 3**: work-first discoveries, dual discovery modes, and feedback-informed exploration
4. **Milestone 4 (in progress)**: curiosity-first discovery, explicit work scope, progress-aware research, and clearer paths into exploration

## Privacy
Library data and SQLite remain local. When you explicitly use Explore Concepts or Surprise Me, relevant titles, notes, URL strings, and concepts are sent to the configured AI provider. API keys remain in backend environment variables and are never sent to the browser or stored in Git.
