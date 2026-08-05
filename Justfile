checks:
	uv run --group dev pre-commit run --all-files

test-fixtures:
	bash assets/resolve-tests-assets.sh

tests:
	uv run --group dev --extra hub --extra backends pytest tests -m 'unit or conformance' --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

tests-unit:
	uv run --group dev --extra hub --extra backends pytest tests -m 'unit' --cov=src --cov-report=term-missing -s -v

tests-conformance:
	uv run --group dev --extra hub --extra backends pytest tests -m 'conformance' --cov=src --cov-report=term-missing --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy -s -v

tests-slow:
	uv run --group dev --extra hub --extra backends pytest tests -m 'slow or integration' --cov=src --cov-report=term-missing --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy -s -v

docs:
	cd docs && uv run --group docs make html
