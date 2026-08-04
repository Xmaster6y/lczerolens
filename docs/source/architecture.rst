Architecture
============

Status and mission
------------------

This page is the normative target for the breaking ``0.5`` refactor.  It
supersedes compatibility-driven extension of the pre-refactor public surface.

**lczerolens runs and instruments Lczero-family evaluators, observes reference
or official search, and turns those observations into reproducible chess
analysis records.**

The design is driven by :doc:`use-cases`.  New public surface must make one of
those workflows possible or materially safer.

System model
------------

The system has two connected planes::

   chess-facing plane

   chess.Board
       |-- LczeroEvaluator.evaluate() --> Evaluation
       |-- analyze_move()/line() ------> MoveAnalysis / LineAnalysis
       |-- Search.run() ---------------> SearchResult + SearchTrace
       `-- counterfactual operator ----> CounterfactualPair
                                              |
                                              v
                                         compare_*()
                                              |
                                              v
                                        DecisionAnalysis

   tensor-execution plane

   chess.Board sequence
       --> stateless Lczero codec
       --> TensorDict inputs
       --> evaluator.model
       --> TensorDict network heads
       --> evaluator transforms / external instrumentation
       --> TensorDict standardized outputs

TensorDict is central to neural execution.  Concrete chess objects are central
to the user-facing analysis API.  Neither representation impersonates the
other.

Public vocabulary
-----------------

``chess.Board``
   The runtime position and history object.  lczerolens does not subclass or
   reimplement it.

``LczeroModel``
   The network-format adapter.  It owns loading, declared native heads, and raw
   network execution, not chess semantics.

``LczeroEvaluator``
   The chess-aware facade that prepares boards, invokes ``LczeroModel``,
   standardizes output tensors, and constructs ``Evaluation`` views.  Its
   ``model`` attribute is the canonical :class:`tensordict.nn.TensorDictModule`
   instrumentation boundary using the documented nested keys.

``Evaluation``
   One position bound to one row of evaluator tensors, with a legal-move policy
   view and explicit native or derived head origins.

``MoveAnalysis`` and ``LineAnalysis``
   Exact python-chess-derived position transitions.  These concrete names are
   preferred over a generic evidence-first interface.

``SearchResult``
   The natural final result shared by reference and official Lczero search.

``SearchTrace``
   The immutable detailed search audit record.  Its available features follow
   from its contents.

``DecisionAnalysis``
   A comparison that relates evaluations, search results, candidates,
   counterfactuals, and exact move analysis without merging their guarantees.

Ownership boundaries
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Owner
     - Responsibility
   * - ``python-chess``
     - Rules, legal moves, outcomes, FEN, variants, and mutable game history.
   * - Lczero
     - Network formats and production engine behavior.  Literal ``lc0`` remains
       only where it is the executable or upstream project's actual identifier.
   * - TensorDict
     - Batched tensor containers, nested keys, device movement, indexing, and
       composable module execution.
   * - ``lczerolens._codec``
     - Lczero input planes, policy vocabulary, move mapping, and legal masks.
   * - ``LczeroModel``
     - Network loading, native head declarations, and raw network execution.
   * - ``LczeroEvaluator``
     - Canonical TensorDict execution keys, chess-aware preparation, and
       standardized evaluation semantics.
   * - lczerolens analysis records
     - Exact facts, move and line effects, validity, provenance, search evidence,
       comparisons, and canonical persistence.
   * - External interpretability packages
     - Hooks, patches, attribution, probes, and intervention methods.
   * - Downstream research
     - Strategic, causal, behavioral, and scientific conclusions.

Canonical TensorDict contract
-----------------------------

The evaluator uses nested keys with stable meanings::

   ("input", "planes")              float [B, 112, 8, 8]
   ("input", "legal_mask")          bool  [B, 1858]

   ("network", "policy_logits")     float [B, 1858]
   ("network", "wdl")               float [B, 3] optional
   ("network", "value")             float [B, 1] optional
   ("network", "mlh")               float [B, 1] optional

   ("evaluation", "policy")         float [B, 1858]
   ("evaluation", "value")          float [B, 1] optional

The corresponding public constants live on ``LczeroKeys`` so integrations do
not duplicate string tuples::

   td[LczeroKeys.INPUT_PLANES]
   td[LczeroKeys.NETWORK_POLICY_LOGITS]
   td[LczeroKeys.EVALUATION_POLICY]

``network`` keys contain exactly what the model emitted.  ``evaluation`` keys
contain standardized or explicitly derived values.  A derived value does not
overwrite a native network head.  ``Evaluation.value`` records whether its
origin was native or derived from WDL.

For a non-terminal position, evaluation-policy entries are zero for illegal
moves and sum to one across legal moves.  For a terminal position the legal
mask and standardized policy are all zero, the policy view is undefined, and
``best_move`` is ``None``.  Raw network heads remain observable.

TensorDict input is validated fail-closed for batch shape, dtype, finite values,
head shape, and required keys.  Instrumentation may add arbitrary nested keys;
the evaluator must preserve them.

Runtime and evidence
--------------------

Runtime objects include modules, TensorDict batches, devices, engine processes,
and mutable search state.  They optimize execution and instrumentation.

Evidence objects include position identity, evaluation records, exact move and
line analysis, counterfactual validity, search traces, and decision analysis.
They are immutable, typed, provenance-bearing, and canonically serializable.

``Evaluation`` is an ergonomic runtime view.  ``Evaluation.record()`` freezes
the selected tensor values and position identity for persistence.  Composite
analysis APIs store records rather than references to live tensors or modules.

Search contract
---------------

Search implementations accept a board and one explicit limit object::

   Search.run(board, Nodes(10_000))
   Search.run(board, Visits(10_000))
   Search.run(board, Simulations(100))
   Search.run(board, Time(milliseconds=500))
   Search.run(board, Depth(20))

Unsupported limits fail before starting the producer.  Requested and observed
work remain separate.

Search trace features are predicates derived from records, not a caller-chosen
maximum capability label.  Root results, root actions, snapshots, events, and
semantic replay can therefore evolve without fabricating unavailable fields.
Accessing absent evidence raises ``SearchEvidenceUnavailable`` with producer
and feature context.

``ReferenceSearch`` is deterministic and auditable.  It is not Lczero-equivalent
and does not own production strength, FPU, batching, collisions, pruning,
transpositions, tree reuse, or time management.  ``LczeroSearch`` invokes an
external engine and translates only evidence that engine actually emitted.

Package structure
-----------------

The target source layout is organized by user action::

   lczerolens/
       model.py
       schema.py
       evaluator.py
       evaluation.py
       moves.py
       counterfactuals.py
       decision.py
       provenance.py
       serialization.py
       search/
           result.py
           trace.py
           reference.py
           lczero.py
           replay.py
       _codec/
           input.py
           policy.py

Dependencies point upward from private codec and model execution toward
evaluation, search, and decision analysis.  Exact chess analysis does not
depend on Torch, TensorDict, datasets, or a model.

Breaking deletion list
----------------------

The release removes rather than deprecates:

* the ``LczeroBoard`` subclass;
* mutable public ``MCTS`` and ``Node``;
* samplers and general self-play;
* generic ``Concept`` and dataset-metric abstractions;
* ``ForceValue`` and flow subclasses;
* generic dataset wrappers;
* board-owned visualization;
* low-level backend helpers that do not implement an evaluator or search
  boundary; and
* legacy notebooks and examples built around removed interfaces.

Feature and release gate
------------------------

A proposed public feature must name the release use case it enables, the
TensorDict or chess-record contract it changes, and the evidence guarantee it
preserves.  Notebook convenience alone is insufficient.

The refactor is releasable only when all use cases pass from an installed wheel,
unit and fixture conformance pass on supported Python versions, live pinned
Lczero conformance passes, documentation is strict-clean, and canonical
artifacts round-trip with stable digests.
