checks:
	uv run --group dev pre-commit run --all-files

tests-assets:
	bash assets/resolve-tests-assets.sh

tests:
	uv run --group dev --extra hub --extra datasets --extra viz --extra backends pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --run-fast --run-backends --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

tests-fast:
	uv run --group dev --extra hub --extra datasets --extra viz --extra backends pytest tests --cov=src --cov-report=term-missing -s -v --run-fast

tests-slow:
	uv run --group dev --group notebooks --extra hub --extra datasets --extra viz --extra backends pytest tests --cov=src --cov-report=term-missing -s -v --run-slow

tests-backends:
	uv run --group dev --extra hub --extra datasets --extra viz --extra backends pytest tests --cov=src --cov-report=term-missing -s -v --run-backends

docs:
	cd docs && uv run --group docs make html
