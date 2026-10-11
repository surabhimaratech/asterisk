# Asterisk ✦

**There's more to the story.**

A personal Curiosity Library for quickly capturing books, movies, podcasts, articles, concepts, and random thoughts. Future AI-powered discovery will surface unexpected, meaningful connections across your interests through **Sidequests**.

## Requirements
- Python 3.10+
- Node.js 20.19+ (or 22.12+)

## Start backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend: http://127.0.0.1:8000

API docs: http://127.0.0.1:8000/docs

## Start frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

The frontend proxies `/api` requests to the backend. Data persists locally in `backend/curiosity.db` (not tracked by git). No accounts or cloud services are required.

## API

- `GET /api/health`
- `GET /api/items?q=&kind=` — search/filter
- `POST /api/items` — add
- `PATCH /api/items/{id}` — edit
- `DELETE /api/items/{id}` — delete

Item kinds: `book`, `movie`, `podcast`, `article`, `concept`, `thought`. Each item has a title, optional note and URL.

## Roadmap
1. **v0.1 (this version)**: capture, list, search, filter, edit, delete
2. AI enrichment: extract mechanisms and concepts while preserving user notes separately
3. Sidequests: generate and critique evidence-backed discovery bridges
4. Autonomous exploration and feedback-driven ranking

## Privacy
This version runs locally. Adding AI services later will require deciding which library content may be sent to external providers.
