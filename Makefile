.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	bash assets/resolve-test-assets.sh

.PHONY: demo-assets
demo-assets:
	bash assets/resolve-demo-assets.sh

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

.PHONY: docs
docs:
	cd docs && poetry run make html

.PHONY: demo
demo:
	poetry run python -m demo.main
