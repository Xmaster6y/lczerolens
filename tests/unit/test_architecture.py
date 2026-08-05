"""Executable package-surface and dependency-direction contract."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

import lczerolens


ROOT = Path(__file__).parents[2]
SOURCE_ROOT = ROOT / "src" / "lczerolens"

EXPECTED_MODULE_DEPENDENCIES = {
    "_decision_serialization": {
        "counterfactuals",
        "decision",
        "facts",
        "moves",
        "provenance",
        "search.result",
        "search.trace",
        "serialization",
    },
    "_codec.input": set(),
    "_codec.policy": {"constants"},
    "behavior": {"_codec", "counterfactuals", "moves", "search.reference", "search.trace"},
    "constants": set(),
    "counterfactuals": {"facts", "moves", "provenance"},
    "decision": {"counterfactuals", "evaluation", "moves", "search.result", "search.trace"},
    "evaluation": {"_codec", "provenance", "schema"},
    "evaluator": {"_codec", "evaluation", "model", "provenance", "schema"},
    "facts": set(),
    "model": set(),
    "moves": {"facts"},
    "provenance": set(),
    "schema": set(),
    "search.lczero": {"search.limits", "search.result", "search.trace"},
    "search.limits": {"search.trace"},
    "search.reference": {
        "_codec",
        "evaluation",
        "evaluator",
        "schema",
        "search.limits",
        "search.result",
        "search.trace",
    },
    "search.result": {"search.trace"},
    "search.trace": {"provenance"},
    "serialization": {"evaluation", "provenance"},
}

EXPECTED_ROOT_EXPORTS = {
    "BehaviorMetric",
    "ChessPlayer",
    "ControlKind",
    "CounterfactualBehaviorComparison",
    "CounterfactualComparison",
    "CounterfactualConstraints",
    "CounterfactualPair",
    "CounterfactualValidity",
    "DecisionAction",
    "DecisionActions",
    "DecisionAnalysis",
    "DecisionAnalysisFormatError",
    "Depth",
    "EvaluatorBehavior",
    "Evaluation",
    "EvaluationBatch",
    "EvaluationProvenance",
    "EvaluationRecord",
    "EvaluationRecordFormatError",
    "Evidence",
    "EvidenceSet",
    "ExactMoveEffect",
    "FactAnalyzer",
    "HistoryPolicy",
    "InputFormat",
    "LczeroEvaluator",
    "LczeroKeys",
    "LczeroModel",
    "LczeroSearch",
    "LineAnalysis",
    "LineAnalysisError",
    "LineFailureReason",
    "LineIntent",
    "LineRole",
    "LineTerminal",
    "MoveAnalysis",
    "Nodes",
    "PositionAttribute",
    "PositionIdentity",
    "ReferenceSearch",
    "RetainedEventReplayCosts",
    "RetainedEventReplayPlan",
    "RetainedEventReplayResult",
    "SearchAction",
    "SearchBehaviorComparison",
    "SearchEvidenceUnavailable",
    "SearchLimit",
    "SearchResult",
    "SearchRoot",
    "SemanticReplayError",
    "SemanticReplayResult",
    "Simulations",
    "Time",
    "Visits",
    "analyze_line",
    "analyze_move",
    "compare_counterfactual",
    "compare_counterfactual_behavior",
    "compare_decision",
    "compare_evaluator_to_search",
    "compare_search_events",
    "evaluator_behavior",
    "plan_retained_events",
    "relocate_piece_counterfactual",
    "remove_piece_counterfactual",
    "replay_retained_events",
    "replay_root_events",
    "replay_search_trace",
    "sibling_counterfactual",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    return ".".join(relative.parts)


def _internal_target(module: str, node: ast.ImportFrom) -> str | None:
    if node.level:
        package = module.rpartition(".")[0]
        package_parts = package.split(".") if package else []
        keep = len(package_parts) - node.level + 1
        parts = package_parts[:keep]
        if node.module:
            parts.extend(node.module.split("."))
        target = ".".join(parts)
    else:
        target = node.module or ""
        if target == "lczerolens":
            return ""
        if target.startswith("lczerolens."):
            target = target.removeprefix("lczerolens.")
    candidate = SOURCE_ROOT / Path(*target.split("."))
    return (
        target if target and (candidate.with_suffix(".py").exists() or (candidate / "__init__.py").exists()) else None
    )


def _runtime_dependencies(path: Path) -> set[str]:
    module = _module_name(path)
    dependencies = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            target = _internal_target(module, node)
            if target:
                dependencies.add(target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("lczerolens."):
                    dependencies.add(alias.name.removeprefix("lczerolens."))
    return dependencies


def test_importable_module_inventory_and_runtime_edges_are_explicit_and_acyclic():
    modules = {_module_name(path): path for path in SOURCE_ROOT.rglob("*.py") if path.name != "__init__.py"}
    assert set(modules) == set(EXPECTED_MODULE_DEPENDENCIES)
    graph = {module: _runtime_dependencies(path) for module, path in modules.items()}
    assert graph == EXPECTED_MODULE_DEPENDENCIES

    def visit(module: str, path: tuple[str, ...]) -> None:
        assert module not in path, f"Runtime import cycle: {' -> '.join((*path, module))}"
        for dependency in graph[module]:
            if dependency in graph:
                visit(dependency, (*path, module))

    for module in graph:
        visit(module, ())


def test_root_exports_are_intentional_unique_and_resolvable():
    assert len(lczerolens.__all__) == len(set(lczerolens.__all__))
    assert set(lczerolens.__all__) == EXPECTED_ROOT_EXPORTS
    assert all(hasattr(lczerolens, name) for name in lczerolens.__all__)


def test_optional_dependencies_stay_outside_the_base_runtime_boundary():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert set(project["project"]["optional-dependencies"]) == {"hub"}
    assert project["dependency-groups"]["conformance"] == ["v-lczero-bindings>=0.31.2"]
    assert "scripts" not in project["dependency-groups"]

    external_imports = []
    for path in SOURCE_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                in {
                    "huggingface_hub",
                    "lczero",
                    "lczero.backends",
                }
            ):
                external_imports.append((_module_name(path), node.module))
            if isinstance(node, ast.Import):
                external_imports.extend(
                    (_module_name(path), alias.name)
                    for alias in node.names
                    if alias.name == "huggingface_hub" or alias.name.startswith("lczero")
                )
    assert external_imports == [("model", "huggingface_hub"), ("model", "huggingface_hub")]
