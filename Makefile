.PHONY: dev backend frontend test lint

dev:
	./scripts/dev.sh

backend:
	python -m uvicorn sentinel.main:create_app --factory --reload --host 127.0.0.1 --port 8765

frontend:
	cd apps/frontend && npm run dev

test:
	pytest apps/backend/tests

lint:
	ruff check apps/backend