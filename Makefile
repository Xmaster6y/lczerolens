# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	bash assets/test-assets.sh

.PHONY: demo-assets
demo-assets:
	bash assets/demo-assets.sh

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

# API
.PHONY: demo
demo:
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
