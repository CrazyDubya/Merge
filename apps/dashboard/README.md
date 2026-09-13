# ChipCliff Dashboard (optional app)

The FastAPI dashboard from **role-based-llm-framework** ("ChipCliff"),
kept verbatim as an optional app per Stephen's decision. It provides the
PM / Coder / Researcher web dashboards and WebSocket UI that front the
role algorithms.

Full source history lives in `repos/role-based-llm-framework/` (merged via
`git subtree`). The role algorithms themselves were also ported onto
club_harness's LLM router in `club_harness/orchestration/roles/`.

## Run

```bash
cd apps/dashboard
pip install -r requirements-dashboard.txt
cp .env.example .env   # add your API keys
uvicorn main:app --reload
```

## Deps

This app is **not** part of Merge core (which stays httpx-only). Its
dependencies are in `requirements-dashboard.txt` (copied from the source
repo's `requirements-minimal.txt`).
