# NeuroAssist (Sprints 1-4)

Implemented state:
- Sprint 1: Flask foundation + DB schema + session creation + MRI upload/inference ingest
- Sprint 2: LLM + RAG foundation + structured report JSON tests
- Sprint 3: Full `session.html` 3-panel clinician UI with live state
- Sprint 4: Standalone report view + print/PDF stub + KB seeder + resilience hardening

## Setup

```powershell
cd e:\Projects\Gioblastoma
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment

`.env` and `.env.example` are configured for:
- host: `localhost`
- port: `5433`
- user: `postgres`
- password: `Admin@123` (URL-encoded in `DATABASE_URL`)

## Database Bootstrap

```powershell
python -m db.init_db
```

Schema behavior:
- Uses `pgvector` when extension exists
- Falls back to `DOUBLE PRECISION[]` embedding column when `pgvector` is unavailable

## Run App

```powershell
python app.py
```

Open: `http://localhost:5000`

## Sprint 4 Features

### Report Delivery
- `GET /report/<session_id>`: standalone structured report view
- `GET /report/<session_id>/print`: print-optimized report route
- "Download report PDF" buttons use browser print flow (`window.print()`) as sprint stub

### Knowledge Base Seeder

```powershell
python -m scripts.seed_knowledge_base
```

Optional append mode:

```powershell
python -m scripts.seed_knowledge_base --append
```

### Reliability / Error Handling
- DB query execution auto-retries once on disconnect (`OperationalError`/`InterfaceError`)
- Groq call path includes exponential backoff on rate limits
- Final report generation returns safe structured errors for:
  - invalid LLM JSON after retry
  - upstream LLM service failures

## Tests

```powershell
python -m unittest tests.test_llm -v
```

Current tests validate LLM JSON shape and strict retry behavior for invalid JSON output.
