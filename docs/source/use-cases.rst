Release use cases
=================

Status
------

This page is the executable product contract for the breaking ``0.5``
refactor.  A public feature belongs in lczerolens only when it is necessary for
one of the use cases below.  The examples define the intended interface; the
refactor is complete when maintained tests run each workflow against the built
wheel.

1. Evaluate a position
----------------------

A user can load an Lczero-family network and inspect its legal move
preferences and available evaluation heads without manually encoding a board
or masking the 1858-entry policy vocabulary::

   import chess
   from lczerolens import LczeroEvaluator

   board = chess.Board()
   evaluator = LczeroEvaluator.from_path("network.onnx")
   evaluation = evaluator.evaluate(board)

   evaluation.policy.best_move
   evaluation.policy["e2e4"].logit
   evaluation.policy["e2e4"].probability
   evaluation.policy.top(5)
   evaluation.wdl
   evaluation.value
   evaluation.mlh

``Evaluation.policy`` is legal-move aware.  Raw network tensors remain
available through ``evaluation.tensors``.  A terminal position has no selected
move and no normalized legal distribution; it may still retain the raw network
heads.

2. Batch and instrument evaluator execution
--------------------------------------------

TensorDict is the canonical neural-execution substrate.  Advanced consumers
can prepare positions, run an instrumented ``LczeroModel``, retain arbitrary
new keys, and finish with the same chess-facing evaluation interface::

   boards = [chess.Board(), chess.Board("8/8/8/8/8/8/4K3/6k1 w - - 0 1")]
   td = evaluator.prepare(boards)

   with external_method.prepare(evaluator.model) as instrumented_model:
       td = instrumented_model(td)

   evaluations = evaluator.finish(boards, td)
   attribution = td["attr", "input", "planes"]

Batch indexing, device movement, nested keys, and composition belong to
TensorDict.  External interpretability tools own the meaning of attribution,
hooks, probes, and interventions.

3. Analyze a move or line exactly
---------------------------------

Exact chess analysis works with a plain :class:`chess.Board` and does not
require an evaluator::

   from lczerolens import analyze_line, analyze_move

   move = analyze_move(board, "e2e4")
   line = analyze_line(board, ["e2e4", "e7e5", "g1f3"])

   move.effects
   move.facts_before
   move.facts_after
   line.steps
   line.final_position
   line.terminal

The results state their guarantee, provenance, and history status.  Exact
effects do not infer strategic concepts such as initiative or compensation.

4. Run reference or official Lczero search
-------------------------------------------

Reference and official Lczero search share one result interface::

   from lczerolens import ReferenceSearch, Simulations

   search = ReferenceSearch(evaluator, c_puct=1.5)
   result = search.run(board, Simulations(100))

   result.move
   result.evaluation
   result.root["e2e4"].visits
   result.root["e2e4"].mean_value
   result.principal_variation
   result.trace

An external engine uses the same shape::

   from lczerolens import LczeroSearch, Nodes

   search = LczeroSearch(
       executable="/path/to/lc0",
       network="network.pb.gz",
       engine_version="v0.31.2",
   )
   result = search.run(board, Nodes(10_000))

``SearchResult`` is the natural final result.  ``SearchTrace`` is the detailed
audit artifact.  Trace features are derived from the evidence actually
present::

   result.trace.has_snapshots
   result.trace.has_events
   result.trace.is_replayable

Requesting unavailable event evidence raises an actionable error.  Root-only
Lczero output is never upgraded to a full or replayable event trace.

5. Compare decisions and counterfactuals
----------------------------------------

The main analysis workflow relates raw evaluator preference, searched
preference, candidate moves, and exact chess evidence::

   from lczerolens import compare_decision

   evaluation = evaluator.evaluate(board)
   search_result = ReferenceSearch(evaluator).run(board, Simulations(100))
   decision = compare_decision(evaluation, search_result)

   decision.policy_move
   decision.search_move
   decision.changed
   decision.actions["e2e4"].policy_rank
   decision.actions["e2e4"].search_rank

Counterfactual construction and model comparison remain separate::

   from lczerolens import compare_counterfactual, sibling_counterfactual

   pair = sibling_counterfactual(board, factual="e2e4", alternative="d2d4")
   comparison = compare_counterfactual(pair, evaluator)
   decision = compare_decision(
       evaluation,
       search_result,
       counterfactuals=[comparison],
   )

   pair.validity
   comparison.policy_change
   comparison.value_change

Structural rule validity never implies historical reachability, and an
observed model difference never implies a causal or strategic conclusion.

6. Persist a reproducible analysis
----------------------------------

Durable evidence records have canonical versioned serialization::

   evaluation_record = evaluation.record()
   evaluation_record.save("evaluation.json")
   restored_evaluation = type(evaluation_record).load("evaluation.json")

   assert restored_evaluation.digest() == evaluation_record.digest()

   decision.save("decision.json")
   restored = type(decision).load("decision.json")

   assert restored.digest() == decision.digest()

The release test exercises evaluation, search, exact move analysis, comparison,
serialization, restoration, and digest stability end to end.  Mutable modules,
engine processes, live tensors, devices, and external instrumentation contexts
are runtime state and are not serialized as evidence.

Explicit non-goals
------------------

The release does not own generic datasets, generic concept metrics, self-play
infrastructure, production search, visualization frameworks, hooks,
attribution, probing, sparse autoencoders, natural-language explanation, or
scientific conclusions.  Examples may integrate such systems through the
TensorDict evaluator boundary.
