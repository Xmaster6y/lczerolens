checks:
	uv run pre-commit run --all-files

tests-assets:
	bash assets/resolve-tests-assets.sh

demo-assets:
	bash assets/resolve-demo-assets.sh

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --run-fast --run-backends --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

tests-fast:
	uv run pytest tests --cov=src --cov-report=term-missing -s -v --run-fast

tests-slow:
	uv run pytest tests --cov=src --cov-report=term-missing -s -v --run-slow

tests-backends:
	uv run pytest tests --cov=src --cov-report=term-missing -s -v --run-backends

docs:
	cd docs && uv run --group docs make html

demo:
	uv run --group demo gradio spaces/demo/app.py

demo-backends:
	uv run --group demo gradio spaces/backends-demo/app.py
