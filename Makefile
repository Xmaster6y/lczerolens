.PHONY: checks
checks:
	uv run pre-commit run --all-files

.PHONY: test-assets
test-assets:
	bash assets/resolve-test-assets.sh

.PHONY: demo-assets
demo-assets:
	bash assets/resolve-demo-assets.sh

.PHONY: tests
tests:
	uv pip install -e .
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --onlyfast

.PHONY: tests-slow
tests-slow:
	uv pip install -e .
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --onlyslow

.PHONY: docs
docs:
	uv pip install -e .
	cd docs && uv run make html

.PHONY: demo
demo:
	uv pip install -e .
	uv run python spaces/lczerolens-demo/app/main.py
