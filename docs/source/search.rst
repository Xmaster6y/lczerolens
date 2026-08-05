Search
======

Natural interface
-----------------

Reference and official Lczero search use the same small interface::

   import chess
   from lczerolens import ReferenceSearch, Simulations

   board = chess.Board()
   result = ReferenceSearch(evaluator).run(board, Simulations(100))

   result.move
   result.evaluation
   result.principal_variation
   result.root["e2e4"].visits
   result.trace

:class:`~lczerolens.search.result.SearchResult` is the final decision-facing
view.  Its ``move`` is always legal at the trace root and agrees with the
producer's final selection.  ``evaluation`` and ``principal_variation`` remain
``None`` when the producer did not expose them.

``result.root`` is a move-keyed immutable
:class:`~lczerolens.search.result.SearchRoot`.  It contains only root-action
fields the producer actually supplied: prior, visits, total and mean value,
exploration, evaluation, leaf evaluation, and principal variation.  Accessing
``root`` when root actions were not exposed raises
:class:`~lczerolens.search.result.SearchEvidenceUnavailable`; lczerolens never
invents plausible statistics to complete a result.

Typed limits
------------

Limits state their unit in the type rather than relying on an ambiguous
integer::

   from lczerolens import Depth, Nodes, Simulations, Time, Visits

   Nodes(10_000)
   Visits(800)
   Simulations(100)
   Time(250)       # milliseconds
   Depth(20)

All count limits are positive integers.  ``Time`` accepts positive finite
milliseconds, although a producer may support only whole milliseconds.
Producers reject unsupported limit types before doing work:

* :class:`~lczerolens.search.reference.ReferenceSearch` supports
  :class:`~lczerolens.search.limits.Simulations`;
* :class:`~lczerolens.search.lczero.LczeroSearch` supports
  :class:`~lczerolens.search.limits.Nodes` and whole-millisecond
  :class:`~lczerolens.search.limits.Time`.

Reference search
----------------

:class:`~lczerolens.search.reference.ReferenceSearch` is a deterministic,
sequential neural-MCTS analysis oracle.  It accepts the same
:class:`~lczerolens.evaluator.LczeroEvaluator` used for direct position
evaluation and returns replayable evidence::

   search = ReferenceSearch(evaluator, c_puct=1.5)
   result = search.run(board, Simulations(100))

The implementation uses
``Q + c_puct * P * sqrt(sum(N)) / (1 + N)``, stable UCI tie-breaking, and
maximum visit count for the final selection.  It intentionally omits Lczero
FPU, batching, virtual visits, collisions, pruning, transpositions, noise,
tree reuse, and time management.  It makes no engine-equivalence or
playing-strength claim.

Reference traces include per-simulation paths, leaf evaluations, expansions,
backups, and pre/post root state.  The public replay helpers independently
check or reconstruct that evidence::

   from lczerolens import replay_root_events, replay_search_trace

   root_statistics = replay_root_events(result.trace.events)
   semantic_result = replay_search_trace(result.trace)

Official Lczero search
----------------------

:class:`~lczerolens.search.lczero.LczeroSearch` invokes a user-supplied
``lc0`` UCI executable and translates only its public root output::

   search = LczeroSearch(
       executable="/path/to/lc0",
       network="network.pb.gz",
       engine_version="v0.31.2",
   )
   result = search.run(board, Nodes(10_000))

The explicit engine version and optional network checksum are evidence
provenance.  User options are retained alongside the requested and observed
budget.  The adapter parses public ``P``, ``N``, ``Q``, ``U``, ``V``, ``WL``,
``D``, and ``PV`` fields when present.  Rounded priors are normalized only when
the complete exposed root set supplies them.

Official output may contain a best move without action statistics.  Such a
result still has a useful ``move`` and trace, but ``result.root`` is
unavailable.  Public root snapshots never imply private engine events, a
complete tree, or replayability.

Trace evidence
--------------

The immutable records live in :mod:`lczerolens.search.trace`.  A
:class:`~lczerolens.search.trace.SearchTrace` retains root identity, producer
and network provenance, requested and observed budgets, snapshots, and any
events the source actually exposed.  Stable UCI strings are used at evidence
boundaries.

Feature properties are derived from contents::

   result.has_root_actions
   result.has_snapshots
   result.has_events
   result.is_replayable

The older ordered :class:`~lczerolens.search.trace.SearchCapability` label and
``supports``/``require`` checks remain on traces for precise existing consumers.
They validate that a producer's claim is no stronger than its records.

Every evaluation and edge statistic names its
:class:`~lczerolens.search.trace.ValuePerspective`.  ``root_player`` means the
side to move in the root FEN; ``side_to_move`` means the player at the position
where the value was recorded.  Consumers must align perspectives before
comparison.  WDL is ordered ``(win, draw, loss)`` and sums to one.  A source
may expose scalar value, WDL, or both; one is not silently derived merely to
fill a trace field.

Search ownership
----------------

Lczero or another chess engine owns production strength, scheduling, tree
reuse, batching, collision handling, pruning, transpositions, and time
management.  lczerolens owns the common result, typed limits, evidence schema,
reference oracle, process translation, and explicit absence semantics.

The pre-refactor mutable search implementation is private compatibility code
used by legacy samplers.  New analysis code should use ``ReferenceSearch`` or
``LczeroSearch`` and consume ``SearchResult``; mutable nodes are not a public
evidence boundary.
