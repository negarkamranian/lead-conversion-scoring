.PHONY: db init ingest quality train evaluate score monitor pipeline mlflow-ui test test-unit lint typecheck format format-fix check build

APP_RUN = docker compose run --rm app uv run --frozen

db:
	docker compose up -d db

build:
	docker compose build app

init: build
	$(APP_RUN) lead-scoring init-db

ingest: build
	$(APP_RUN) lead-scoring ingest

quality: build
	$(APP_RUN) lead-scoring quality

train: build
	$(APP_RUN) lead-scoring train

evaluate: build
	$(APP_RUN) lead-scoring evaluate

score: build
	$(APP_RUN) lead-scoring score

monitor: build
	$(APP_RUN) lead-scoring monitor

pipeline: build
	$(APP_RUN) lead-scoring pipeline

mlflow-ui: build
	docker compose run --rm --no-deps -p 5000:5000 app uv run --frozen mlflow server --backend-store-uri sqlite:////app/mlruns/mlflow.db --default-artifact-root /app/mlruns/artifacts --host 0.0.0.0 --port 5000

test: build
	$(APP_RUN) pytest -q

test-unit: build
	$(APP_RUN) pytest -q -m 'not integration'

lint: build
	$(APP_RUN) ruff check .

format: build
	$(APP_RUN) ruff format --check .

format-fix: build
	$(APP_RUN) ruff format .

typecheck: build
	$(APP_RUN) mypy

check: format lint typecheck test
