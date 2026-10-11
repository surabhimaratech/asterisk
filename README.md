# Asterisk ✦

**There's more to the story.**

A personal Curiosity Library for quickly capturing books, movies, podcasts, articles, concepts, questions, and random thoughts. Future AI-powered discovery will surface unexpected, meaningful connections across your interests through **Sidequests**.

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

The frontend proxies `/api` requests to the backend. Data persists at `backend/data/asterisk.db`, using an absolute path derived from the application location rather than the terminal's working directory. Asterisk creates the data directory automatically, and `*.db` files are excluded from git. No accounts or cloud services are required.

If an earlier installation has `backend/curiosity.db`, Asterisk copies it to the new location on first startup when `backend/data/asterisk.db` does not yet exist. The original file is left untouched.

## API

- `GET /api/health`
- `GET /api/items?q=&kind=` — search/filter
- `POST /api/items` — add
- `PATCH /api/items/{id}` — edit
- `DELETE /api/items/{id}` — delete

Item kinds: `book`, `movie`, `podcast`, `article`, `concept`, `question`, and `thought` (shown as “Random thought” in the interface). Each item has a title, optional personal note and source URL, and an automatic creation timestamp.

## Tests

```bash
backend/.venv/bin/pytest

cd frontend
npm run build
```

## Roadmap
1. **v0.1 (this version)**: capture, list, search, filter, edit, delete
2. AI enrichment: extract mechanisms and concepts while preserving user notes separately
3. Sidequests: generate and critique evidence-backed discovery bridges
4. Autonomous exploration and feedback-driven ranking

## Privacy
This version runs locally. Adding AI services later will require deciding which library content may be sent to external providers.
