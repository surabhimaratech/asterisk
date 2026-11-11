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
- `GET /api/discovery-profile` — inspect working hypotheses derived from explicit positive feedback
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
- `POST /api/discoveries/research` — run the Discovery Engine beta for one selected item
- `GET /api/discoveries/saved` — list saved Little Discoveries
- `POST /api/discoveries/saved` — save an array of discovery snapshots
- `DELETE /api/discoveries/saved/{candidate_id}` — remove a saved discovery

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

Each grounded observation separately records the concrete detail, the mechanism or dynamic that makes it interesting, a specific curiosity invitation, its appeal dimensions, and whether it describes one work, a creator/series pattern, a genre convention, or a possible cross-domain direction. Generic curiosity questions are rejected before persistence. These labels describe epistemic and discovery roles; they do not upgrade an interpretation into a fact.

The selected observations support two discovery modes: **deepen fascination**, which follows something already compelling in the work, and **reveal an overlooked perspective**, which surfaces a less obvious lens. External research still uses bounded OpenRouter work research, evidence-grounded concept extraction, and relationship validation. Search-tool charges and ordinary model token charges apply. If external research fails, Asterisk does not invent a concept; the separately work-supported observations remain available. Research permits full spoilers only when the item is marked finished/re-reading/rewatching or the user explicitly allows them.

Extraction scans the full independent research corpus, not only passages that answer the first research problems. Accepted results distinguish **From the work** (`work_grounded`) from **Found while exploring** (`research_led`). Research-led concepts need independent definition and mechanism evidence plus a traceable research path, but may honestly report that no direct relationship to the saved work was established.

When a promising candidate fails only because its evidence cannot be verified, Asterisk first tries to rebind an exact excerpt already present elsewhere in the corpus, then performs one bounded semantic review of that same corpus. It may then make one targeted retrieval and evidence-repair attempt for a strictly limited number of candidates. Every proposed passage is still checked verbatim by the backend. `EVIDENCE_RECOVERY_MAX_CANDIDATES` (default `2`), `EVIDENCE_RECOVERY_MAX_SEARCH_RESULTS` (default `4`), and the estimated-spend settings in `.env.example` bound this extra work. Recovery never lowers the quality threshold, never retries automatically, and records the initial failure, recovery action, and final outcome for inspection. The spend limit is an internal estimate rather than a provider-enforced billing cap; actual search and token prices remain provider-dependent.

Concept extraction is always user-triggered. Quick extraction uses only the saved entry and is labeled accordingly. Research results distinguish **Established concept**, **Grounded interpretation**, and **Speculative connection**. A generated interpretive label does not need to appear verbatim when its underlying mechanism and relationship are supported. Actual quotations remain subject to exact source-text checks; clearly labeled cited paraphrases may use different wording but still require a retrieved, substantive source and structured validation. Speculative connections remain in review unless the user explicitly keeps them.

Rejected concept candidates can be reviewed in the Research & Concepts drawer. **Interested**, **Love**, and **Surprise** are independent from **Keep anyway**, **Dismiss**, and **Re-evaluate**: they record personal discovery value without asserting truth. Feedback is copied across reruns only when a stable fingerprint of the underlying finding matches; similar titles alone are not enough. Keep anyway creates a non-canonical discovery labeled **User-kept · Unverified** or **User-kept · Partially supported** without changing the automated rejection. Reversible review history and evidence outcomes remain intact. User-kept concepts are available to exploratory features, while their sources are deliberately withheld as claim-supporting evidence and they remain outside verified-only workflows.

A lightweight curiosity profile is derived locally from active Love, Surprise, and Interested signals, their optional notes, and user-authored concepts. It is a transparent working hypothesis—not a permanent taste label. It can modestly influence anchor ordering, but cannot act as evidence, infer dislike from silence or dismissal, or consume the reserved unfamiliar-direction slot.

Discovery cards lead with the observation, mechanism, and “Where could this lead?” invitation. Love, Surprise, and Interested remain separate, keyboard-accessible actions; an optional context disclosure lets you explain a reaction without adding friction to the default flow. Evidence, provenance, validation quality, and recovery diagnostics are available under Research & provenance rather than dominating the reading experience. Short personal concepts use a compact card without empty generated sections.

Specific “Where could this lead?” questions can be saved as exploration invitations. These are persistent, reversible investigation seeds shown in the Sidequests area. They modestly influence candidate ordering and are passed to generation as user-selected questions, never as factual evidence or as a claim that an investigation has already happened. An exact regenerated discovery reconnects through its stable feedback key; a retired or removed source remains visible but unavailable until that discovery returns.

Sidequests can open a direction from one prepared item—including a random thought—or connect multiple library items. Each manual run reconsiders a bounded, ranked sample from the eligible library. Interest, Love, Surprise, saved invitations, mechanism depth, grounding, diversity, and repetition avoidance shape that sample without upgrading verification; an exploration slot prevents familiar interests from permanently crowding out other domains. Stable origin-and-content keys reject repeated results across runs. Every saved Sidequest remains exploratory and externally unverified.

Detective Mode begins with one concrete clue and follows it either toward an unfamiliar subject or, only when a real mechanism warrants it, into another library item. Within the same generation call, the strongest clue produces competing outward directions across different domains; at most one optional library connection joins that competition. The backend requires at least two distinct external directions from the same clue before critique, comparing their actual lead identity and investigative substance rather than trusting the declared category. The critic then assesses every surviving candidate for lead alignment, a concrete causal/structural/historical/artistic/cultural/emotional/formal/conceptual mechanism, and a next investigation that actually pursues the lead. An external label wrapped around a library comparison therefore cannot win, while a distinctive library connection can still compete after outward competition exists. Candidate dossiers are bounded to one focus finding, a small optional catalog of library leads, and two current excerpts per finding; `SIDEQUEST_MAX_CONTEXTS` limits the number of clue-first contexts sent to the model. The existing two-call generation/critique budget is unchanged. Leads name a concrete phenomenon, practice, historical event, intellectual problem, artistic technique, scientific mechanism, or library curiosity. The selected case file separates sourced facts, interpretations, and speculative hypotheses. Only supported facts require eligible evidence; an unsourced unfamiliar subject can remain an explicitly open lead. Deterministic checks reject invented source references and self-described abstract/theme-only comparisons. When competition or semantic alignment is insufficient, the API returns “No compelling case yet” and saves nothing. Additive JSON fields preserve the clue, lead, short reasoning trail, and claims without rewriting legacy Sidequests.

Each manual generation emits structured `sidequest_diagnostic` application logs with a short run ID. They appear in the backend terminal at INFO level when using `./scripts/dev.sh`; set `ASTERISK_LOG_LEVEL` to override the application and Uvicorn level. The `contexts`, `generation`, `validation`, `competition`, `critic`, `critic_assessment`, and `final` records show counts, grouped validation reasons, distinct and duplicate external leads, semantic mismatch counts, the critic outcome, and whether anything was saved. They intentionally omit clue text, notes, source excerpts, and API credentials.

### Discovery Engine v0.4 — Little Discoveries

The separate **Research a Sidequest (Beta)** experiment starts from one item chosen by the user. It proposes two or three searchable directions in one structured model call, performs one bounded OpenRouter web-research request (no more than three descriptive queries and `DISCOVERY_ENGINE_MAX_RESULTS`, default 6, total results), then uses one structured synthesis call to produce a standalone discovery. Factual claims must cite opaque IDs from the retrieved source set; the API returns the exact retrieved URLs and excerpts rather than model-authored citations. Interpretations and speculation remain separately labeled, and a library connection is optional.

Within the same final synthesis request, Asterisk shapes up to three supported particulars into little stories: a specific hook, a two-to-four sentence account, the evidence-grounded turn that makes it worth noticing, an optional thread back to the library, and a concrete way to go deeper. Multiple stories may come from one productive research direction. A broad conclusion or clever title cannot substitute for a documented particular, and every factual detail must be supported by the candidate's retrieved source IDs.

The API returns up to three conservatively deduplicated `discovery_candidates` from that same synthesis call. Candidates with missing or unknown source IDs are omitted, and each displayed source is resolved from the actual bounded retrieval packet rather than a model-authored URL. The story stays in the foreground; research direction, evidence assessment, and bounded excerpts live behind **Sources & details**. The UI identifies Asterisk's suggested pick while letting the user explore, hide a story for the current run, or save it for later. Saved discoveries can be reopened with their source links and follow-up prompts. Only the valid suggested candidate receives the full synthesized discovery card. Saving and removing discoveries use the local database without making provider requests.

A research run returns its results to the current browser view. Explicitly saved stories are stored in the local SQLite database and available to browsers connected to that Asterisk instance. On opening Little Discoveries, Asterisk imports any valid saves from the previous browser-only version and clears that browser copy after confirmation. A normal successful run therefore makes three provider requests: direction generation, research with web search, and evidence-bound synthesis. Empty research or insufficient evidence returns a readable no-discovery response without changing library or Sidequest data.

Beta diagnostics are disabled by default. For one inspectable local run, add `DISCOVERY_DEBUG=1` to `.env`, restart `./scripts/dev.sh`, and watch the backend terminal for `discovery_debug` JSON records sharing one `run_id`. The `seed`, `directions`, `research_request`, `sources`, `synthesis`, and `outcome` stages show where a result narrowed or repeated. Logs include only the seed title and concept names, generated directions, submitted descriptive queries, bounded source excerpts, and the structured synthesis. Personal notes and raw prompts are omitted; credential-shaped text is redacted, and source URL credentials, query strings, and fragments are removed. OpenRouter does not expose the exact provider-executed web-search queries in its response, so that field is explicitly recorded as unavailable. Remove the setting or set it to `0` and restart to disable diagnostics again.

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
