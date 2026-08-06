Search
======

Run :doc:`notebooks/features/replayable-search` for the executable workflow.

One result interface
--------------------

Reference and official Lczero search return ``SearchResult``:

.. code-block:: python

   result = ReferenceSearch(evaluator).run(board, Simulations(100))

   result.move
   result.root["e2e4"].visits
   result.principal_variation
   result.trace

Limits name their unit: ``Simulations``, ``Nodes``, ``Visits``, ``Time``, or
``Depth``. Producers reject unsupported limits before starting work.

Evidence capabilities
---------------------

A result exposes only evidence supplied by its producer. Check
``has_root_actions``, ``has_snapshots``, ``has_events``, or ``is_replayable``
before consuming an optional feature. Accessing unavailable evidence raises
``SearchEvidenceUnavailable``; lczerolens does not invent missing statistics.

Every value states its perspective. Consumers must align root-player and
side-to-move values before comparison.

Search producers
----------------

``ReferenceSearch`` is deterministic, sequential, and replayable. It is an
analysis oracle, not an lc0-equivalent or production-strength search. Its
``c_puct`` vocabulary follows the UCT and policy-guided search lineage
:cite:p:`kocsis2006,silver2018`; this does not make it an AlphaZero or lc0
reproduction. Semantic and retained-event replay apply to its full event
evidence.

``audit_search_trace`` returns a reconstructed root checkpoint after every
event (and marks a stopped divergent event incomplete), cumulative
node/expansion/evaluator-call counts, raw field-level
recorded-versus-replayed discrepancies, the first divergence, and the explicit
``ReplayTolerance`` used for numerical comparisons. ``replay_retained_events``
also reports the exact nodes, edges, paths, expansions, evaluator calls, costs,
and ancestor-closure status induced by the selected events. It never inserts
omitted events or backup contributions; selectors, fidelity metrics, and pass
thresholds remain downstream concerns.

``LczeroSearch`` invokes a supplied lc0 executable :cite:p:`lczero` and
translates its public root output. Public root snapshots do not imply private
events, a complete tree, or replayability.

Live engine validation is explicit:

.. code-block:: console

   LC0_EXECUTABLE=/path/to/lc0 \
   LC0_NETWORK=/path/to/network.pb.gz \
   LC0_VERSION=v0.31.2 \
   just tests-live-lczero

The normal suite remains hermetic and uses captured or mocked engine output.
