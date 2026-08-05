Tutorials
=========

All notebooks use deterministic fixtures and run offline during the docs build.

Feature notebooks
-----------------

* :doc:`notebooks/features/models-and-inputs` loads a local model, encodes
  position batches, moves execution to the available device, and preserves an
  instrumentation key through ``finish``.
* :doc:`notebooks/features/evaluate-positions` inspects legal policy, scalar
  evaluation, batching, and immutable evaluation records.
* :doc:`notebooks/features/chess-evidence` keeps exact facts, move/line
  analysis, counterfactual validity, and authored puzzle correctness distinct.
* :doc:`notebooks/features/replayable-search` runs deterministic reference
  search, inspects capabilities, and replays semantic and retained-event
  evidence. It also states the official-engine boundary.

Tutorial notebooks
------------------

* :doc:`notebooks/tutorials/decision-analysis` composes evaluator, search,
  exact line, counterfactual, puzzle, and serialization records end to end.
* :doc:`notebooks/tutorials/compare-models` compares two pinned evaluator
  producers across the same positions without turning disagreement into a
  causal or strength claim.
* :doc:`notebooks/tutorials/analyze-puzzles` grades branching authored
  solutions, analyzes an accepted line, and keeps model preference separate
  from puzzle correctness.

.. toctree::
   :hidden:
   :maxdepth: 2

   decision-analysis-tutorial
   notebooks/features/models-and-inputs
   notebooks/features/evaluate-positions
   notebooks/features/chess-evidence
   notebooks/features/replayable-search
   notebooks/tutorials/decision-analysis
   notebooks/tutorials/compare-models
   notebooks/tutorials/analyze-puzzles
