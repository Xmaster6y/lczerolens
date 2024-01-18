# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s

# API
.PHONY: app-start
app-start:
	poetry run python -m demo.main

# Docker
.PHONY: docker-build
docker-build:
	docker compose -f docker/docker-compose.yml build

.PHONY: docker-start
docker-start:
	docker compose -f docker/docker-compose.yml up

.PHONY: docker-start-bg
docker-start-bg:
	docker compose -f docker/docker-compose.yml up -d --build

.PHONY: docker-stop
docker-stop:
	docker compose -f docker/docker-compose.yml down

.PHONY: docker-tty
docker-tty:
	docker compose -f docker/docker-compose.yml exec fastapi bash
