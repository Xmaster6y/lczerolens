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
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --onlyfast --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

.PHONY: tests-slow
tests-slow:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --onlyslow --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

.PHONY: docs
docs:
	cd docs && uv run make html

.PHONY: demo
demo:
	uv run python spaces/lczerolens-demo/app/main.py
