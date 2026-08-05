Features
========

The maintained documentation describes the supported library surface through
concrete, tested workflows.

* :doc:`scope` defines board encoding, evaluator outputs, model formats, and
  the external-integration boundary.
* :doc:`facts` covers exact position facts, move and line analysis, and
  constrained counterfactuals.
* :doc:`search` defines provenance, capabilities, snapshots, and the
  deterministic reference-search boundary.
* :doc:`use-cases` defines evaluator, authored-puzzle, counterfactual, search,
  and decision compositions through concrete chess records.

The old notebook collection depended on mutable MCTS, samplers, generic
datasets, and concept abstractions. Those examples were removed with those
interfaces instead of being presented as supported workflows.
