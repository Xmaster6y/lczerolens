checks:
	uv run --group dev pre-commit run --all-files

test-fixtures:
	bash assets/resolve-tests-assets.sh

tests:
	uv run --group dev --group conformance --extra hub pytest tests -m 'unit or conformance' --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

tests-unit:
	uv run --group dev --extra hub pytest tests -m 'unit' --cov=src --cov-report=term-missing -s -v

tests-conformance:
	uv run --group dev --group conformance --extra hub pytest tests -m 'conformance' --cov=src --cov-report=term-missing --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy -s -v

tests-slow:
	uv run --group dev --extra hub pytest tests -m 'slow or integration' --cov=src --cov-report=term-missing --cov-branch --cov-report=xml --junitxml=junit.xml -o junit_family=legacy -s -v

tests-wheel:
	LCZEROLENS_RUN_WHEEL_TEST=1 uv run --group dev pytest -q tests/unit/test_distributions.py

tests-live-lczero:
	#!/usr/bin/env bash
	set -euo pipefail
	: "${LC0_EXECUTABLE:?set LC0_EXECUTABLE to the pinned lc0 binary}"
	: "${LC0_NETWORK:?set LC0_NETWORK to the pinned network file}"
	: "${LC0_VERSION:?set LC0_VERSION to the exact output version token}"
	uv run --group dev pytest -q tests/unit/test_lczero_search.py::test_optional_pinned_lczero_process_adapter

docs:
	cd docs && uv run --group docs make html
