Release use cases
=================

The maintained notebooks are the executable product contract:

1. Load models and inspect canonical inputs —
   :doc:`notebooks/features/models-and-inputs`.
2. Evaluate positions and batches —
   :doc:`notebooks/features/evaluate-positions`.
3. Analyze exact moves, lines, counterfactuals, and authored tasks —
   :doc:`notebooks/features/chess-evidence`.
4. Run reference search and consume only available evidence —
   :doc:`notebooks/features/replayable-search`.
5. Compare evaluator and search decisions and persist the result —
   :doc:`notebooks/tutorials/decision-analysis`.
6. Compare two evaluators on identical positions —
   :doc:`notebooks/tutorials/compare-models`.
7. Grade and analyze an authored puzzle collection —
   :doc:`notebooks/tutorials/analyze-puzzles`.

Required guarantees
-------------------

* ``chess.Board`` owns chess rules and history.
* ``LczeroModel`` owns network execution; ``LczeroEvaluator`` owns encoding,
  legal policy, and standardized heads.
* TensorDict is the neural execution boundary; immutable records are the
  persistence boundary.
* Exact chess evidence stays separate from heuristic interpretation.
* Search traces never claim evidence a producer did not expose.
* Puzzle correctness comes only from the authored solution tree.
* Canonical serialization revalidates restored evidence.

The project does not own production search, generic datasets, attribution,
probing, sparse autoencoders, visualization frameworks, natural-language
coaching, or scientific conclusions. Those systems may integrate at the
documented boundaries.
