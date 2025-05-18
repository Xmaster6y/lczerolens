.PHONY: checks
checks:
	uv run pre-commit run --all-files

.PHONY: tests-assets
tests-assets:
	bash assets/resolve-tests-assets.sh

.PHONY: demo-assets
demo-assets:
	bash assets/resolve-demo-assets.sh

.PHONY: tests-fast
tests-fast:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --onlyfast --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

.PHONY: tests-slow
tests-slow:
	uv run pytest tests --cov=src --cov-report=term-missing -s -v --onlyslow

.PHONY: tests-backends
tests-backends:
	uv run pytest tests --cov=src --cov-report=term-missing -s -v --onlybackends

.PHONY: docs
docs:
	cd docs && uv run --group docs make html

.PHONY: demo
demo:
	uv run --group demo gradio spaces/lczerolens-demo/app.py

.PHONY: demo-backends
demo-backends:
	uv run --group demo gradio spaces/lczerolens-backends-demo/app.py
