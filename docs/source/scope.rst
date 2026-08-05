Scope and compatibility policy
==============================

Mission
-------

**lczerolens makes lc0-family models portable and operable in PyTorch, then
expresses their evaluator and search behavior as chess-domain evidence.**

The two halves are mutually necessary: a chess analysis cannot be reproduced or
compared without a reliable lc0 evaluator contract, and model outputs become
useful chess evidence only when they are related to positions, legal moves,
variations, and search decisions.

Supported use cases
-------------------

The currently shipped public surface includes:

* loading converted lc0-family weights and evaluating one position or a batch
  in PyTorch;
* encoding positions and mapping policy indices to legal chess moves;
* consuming a standardized policy, WDL, value, and MLH evaluator result;
* passing an evaluator through arbitrary external instrumentation while
  retaining the same input/output contract.
* recording exact position facts and legal move/line analysis;
* constructing constrained sibling, structural, and piece-relocation
  counterfactual records with explicit validity and reachability guarantees;
* producing typed search traces from deterministic reference search or exposed
  lc0 root snapshots; and
* composing evaluator, counterfactual, and search evidence into one concrete
  decision analysis without upgrading a source's available evidence.

Search records follow the capability-aware schema in :doc:`search`.
Counterfactual and decision-comparison facilities are merged public APIs, with
their guarantees and limitations defined by :doc:`facts` and :doc:`use-cases`.

Evaluator contract
------------------

The chess boundary is a plain ``chess.Board`` position in and an ``Evaluation``
out. ``LczeroEvaluator.prepare`` encodes a board batch and its legal mask into
the canonical ``TensorDict`` keys; ``evaluator.model`` performs raw network
execution; and ``LczeroEvaluator.finish`` validates and standardizes the heads
without discarding instrumentation keys. ``LczeroModel`` owns loading and raw
TensorDict execution only. Legal masking and normalization belong to the
evaluator rather than to a mutable board subclass.

Model-format compatibility
--------------------------

``LczeroModel.from_path`` supports ``.onnx`` files through ``onnx2torch`` and
serialized ``.pt`` ``torch.nn.Module`` objects. Official lc0 weights can be
converted externally with the ``leela2onnx`` command supplied by lc0, then
loaded from the resulting ONNX file. Native bindings remain a test-only
conformance oracle; ordinary PyTorch inference and unit tests do not require
them.
Arbitrary PyTorch modules can be wrapped with ``LczeroModel(module, out_keys)``;
automatic head discovery is reserved for converted lc0 graphs and reports how
to provide explicit keys when that structure is unavailable.

Architecture boundary
---------------------

.. code-block:: text

   python-chess                         external interpretability tools
   rules, FEN, legal moves              hooks, patches, probes, attribution
          |                                            |
          v                                            | wrap or instrument
   chess.Board -- stateless codec --> TensorDict / PyTorch evaluator
                                              conversion, loading, TensorDict batching
          |
          v
   evaluator contract -- policy / WDL / value / MLH
          |
          +-----------------------------+
          |                             |
          v                             v
   chess-semantic analysis          search traces and comparisons
   position facts, move/variation,  evaluator-guided MCTS evidence
   counterfactual evidence

``python-chess`` owns chess rules. External tools own neural-method semantics.
lczerolens owns the arrows between them: lc0 interoperability, the stable
evaluator result, and chess-specific decision evidence. Search is an analysis
and comparison facility, not a production engine.

Non-goals
---------

The core API does not own chess rules, production-engine search, generic hook
or patch systems, attribution algorithms, probing, SAE/transcoder methods, or
natural-language coaching. Examples may demonstrate external tools, but no
implementation-specific interpretability method is required by the core API.

Public-surface disposition
--------------------------

The table covers every package responsibility. ``Retain`` means supported as
part of the stated boundary. ``Internalize`` means implementation-facing rather
than a separate compatibility commitment.

.. list-table::
   :header-rows: 1
   :widths: 14 38 12 36

   * - Module
     - Symbols
     - Disposition
     - Rationale
   * - ``__init__``
     - Model/evaluator imports plus facts, counterfactual, reference search,
       move-evidence, and decision-analysis entry points
     - Retain
     - Stable convenience entry point for the documented chess-decision surface.
   * - ``_codec``
     - Stateless input/policy helpers
     - Internalize
     - Private Lczero transport used by the public evaluator and search boundaries.
   * - ``constants``
     - Policy-index and encoding constants
     - Internalize
     - Authoritative substrate; avoid promising incidental names.
   * - ``model``
     - ``LczeroModel``
     - Retain
     - Raw TensorDict execution and model-format loading beneath the evaluator;
       no head-filtering flow hierarchy or hook API.
   * - ``schema``
     - ``LczeroKeys``
     - Retain
     - One stable vocabulary for the nested TensorDict execution contract.
   * - ``evaluator``
     - ``LczeroEvaluator``, ``InputFormat``
     - Retain
     - Natural board preparation, TensorDict execution, head validation, and
       chess-semantic finishing boundary.
   * - ``evaluation``
     - ``Evaluation``, ``EvaluationBatch``, ``EvaluationRecord`` and their
       typed policy/value/WDL views
     - Retain
     - Separate ergonomic runtime tensors from immutable, reproducible evidence.
   * - ``provenance``
     - ``ChessPlayer``, ``PositionIdentity``, ``EvaluationProvenance``
     - Retain
     - Reconstructable board history and explicit producer/network identity.
   * - ``serialization``
     - Canonical evaluation-record encoding helpers
     - Internalize
     - Persistence is naturally exposed by ``EvaluationRecord`` methods; the
       wire-format machinery is not a second user workflow.
   * - ``search``
     - ``SearchResult``, typed limits, ``ReferenceSearch``, ``LczeroSearch``,
       trace and replay records
     - Retain
     - One natural result contract with explicit producer limits and absence
       semantics. Reference search is replayable; official public output is
       root-only. Mutable MCTS, samplers, and self-play are not part of the
       release surface.
   * - ``facts``
     - Evidence records, guarantees, analyzers, and ``FactAnalyzer``
     - Retain
     - Exact chess observations with provenance before a consumer derives labels or claims.
   * - ``moves``
     - ``MoveAnalysis``, ``LineAnalysis``, ``analyze_move``, ``analyze_line``
     - Retain
     - Legal-move and ordered-line analysis built from exact facts.
   * - ``counterfactuals``
     - Constraints, validity/result records, and sibling/structural operators
     - Retain
     - Constrained position pairs that state rule validity and historical reachability separately.
   * - ``decision``
     - ``DecisionAnalysis``, action records, and evaluator/search/counterfactual
       comparison helpers
     - Retain
     - Compose already-produced evidence without merging its guarantees.

Compatibility policy
--------------------

For the breaking ``0.5`` release, plain ``chess.Board``, ``LczeroEvaluator``,
``LczeroModel``, ``InputFormat``, and the evaluator field meanings form the compatibility core.
The private codec may evolve without becoming a second public board API.

New feature gate
----------------

Before adding public surface, a proposal must identify which supported use case
it advances, state its evaluator-contract effect, and show why it belongs to
lc0 interoperability or chess-decision evidence. If it instead owns a generic
interpretability technique, it belongs in an external integration or example.
