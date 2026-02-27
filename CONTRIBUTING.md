# Contributing

## Prerequisites

- Python 3.11+
- Node.js 20+
- npm 10+

## Setup

```bash
pip install -e .[dev]
cd apps/frontend && npm install
```

## Dev Run

- macOS/Linux: `./scripts/dev.sh`
- Windows PowerShell: `./scripts/dev.ps1`

## Quality Gates

Backend:
- `ruff check apps/backend`
- `mypy apps/backend/sentinel`
- `pytest apps/backend/tests`

Frontend:
- `npm run lint`
- `npm run typecheck`
- `npm run build`

## PR Checklist

- Tests added/updated for behavior changes.
- Security-sensitive fields are redacted.
- Docs updated when user-visible behavior changes.
- No runtime writes to repo directory.
- Accessibility and responsive behavior validated.