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

Current v0.4 capabilities are:

* loading or converting lc0-family weights and evaluating one position or a
  batch in PyTorch;
* encoding positions and mapping policy indices to legal chess moves;
* consuming a standardized policy, WDL, value, and MLH evaluator result;
* passing an evaluator through arbitrary external instrumentation while
  retaining the same input/output contract.

The planned in-scope public surface will add chess-domain evidence for position
facts, moves, variations, counterfactual positions, and search traces and
comparisons. Those facilities do not yet have a counterfactual API or a
search-trace schema in v0.4; this policy defines their ownership before they are
introduced.

Evaluator contract
------------------

Conceptually, the chess boundary is an ``LczeroBoard`` position in and a
``TensorDict`` evaluator result out. At runtime, ``LczeroModel.forward()`` also
accepts an iterable of boards, a 3D or 4D board tensor, or a ``TensorDict``.
The output contains raw policy values over the fixed 1858-entry lc0 vocabulary
and WDL probabilities when the network supplies them; value and MLH are exposed
when supplied or derived by an explicit adapter such as ``ForceValue``.
Batching and device movement preserve this contract. Legal masking is a
downstream operation: the wrapper returns raw policy values, and consumers must
select the board's legal indices before interpreting a move choice.
``LczeroBoard.get_legal_policy(policy)`` performs that selection and softmax
normalization for one policy vector; it rejects terminal positions, malformed
vectors, and non-finite legal logits rather than returning an invalid
distribution.

Model-format compatibility
--------------------------

``LczeroModel.from_path`` supports ``.onnx`` files through ``onnx2torch`` and
serialized ``.pt`` ``torch.nn.Module`` objects. Official lc0 weights are
converted with the optional native backend's ``leela2onnx`` command, then loaded
from the resulting ONNX file. Native bindings are a conversion and conformance
oracle only; ordinary PyTorch inference and unit tests do not require them.
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
   LczeroBoard -- encoding / move vocabulary --> lc0 adapters / PyTorch evaluator
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

The table covers every current importable package module and its public symbols.
``Retain`` means supported as part of the stated boundary; ``refocus`` means the
symbol stays available but future changes must serve that boundary. ``Internalize``
means keep it implementation-facing rather than adding compatibility commitments;
``deprecate`` means do not extend it and replace it in a later, separately
announced compatibility release.

.. list-table::
   :header-rows: 1
   :widths: 14 38 12 36

   * - Module
     - Symbols
     - Disposition
     - Rationale
   * - ``__init__``
     - ``LczeroBoard``, ``LczeroModel``
     - Retain
     - Small stable entry point for positions and evaluators.
   * - ``board``
     - ``InputEncoding``, ``LczeroBoard``
     - Retain
     - lc0 encoding, legal moves, and policy vocabulary bridge.
   * - ``constants``
     - Policy-index and encoding constants
     - Internalize
     - Authoritative substrate; avoid promising incidental names.
   * - ``model``
     - ``LczeroModel``, ``ForceValue``; ``Flow``, ``PolicyFlow``,
       ``ValueFlow``, ``WdlFlow``, ``MlhFlow``
     - Retain; refocus flows
     - Preserve evaluator loading and output adapters; do not make a hook API.
   * - ``backends``
     - ``generic_command``, ``describenet``, ``convert_to_onnx``,
       ``convert_to_leela``, ``board_from_backend``,
       ``prediction_from_backend``, ``moves_with_castling_swap``
     - Refocus
     - lc0 conversion/backend interoperation; low-level helpers remain subordinate to model adapters.
   * - ``data``
     - ``columns_to_rows``, ``rows_to_columns``, ``GameData``,
       ``BoardData``, ``PuzzleData``
     - Refocus
     - Position/game/puzzle evidence adapters, not a general data platform.
   * - ``concepts``
     - ``Concept``, ``BinaryConcept``, ``NullConcept``, ``OrBinaryConcept``,
       ``AndBinaryConcept``, ``MulticlassConcept``, ``ContinuousConcept``,
       ``HasPiece``, ``HasMaterialAdvantage``, ``BestLegalMove``,
       ``PieceBestLegalMove``, ``HasThreat``, ``HasMateThreat``
     - Refocus
     - Chess-semantic position and move facts; no generic probing framework.
   * - ``sampling``
     - ``Sampler``, ``RandomSampler``, ``ModelSampler``, ``PolicySampler``,
       ``MCTSSampler``, ``SelfPlay``
     - Refocus
     - Evaluation-driven move and self-play comparisons, not engine serving.
   * - ``search``
     - ``Heuristic``, ``RandomHeuristic``, ``MaterialHeuristic``,
       ``ModelHeuristic``, ``Node``, ``MCTS``
     - Refocus
     - Search traces and comparisons supporting decision evidence.

Compatibility policy
--------------------

The top-level ``LczeroBoard`` and ``LczeroModel`` imports, the board encoding
and policy-index mapping, and the evaluator field meanings are the compatibility
core. A compatible release must preserve their documented semantics or provide a
deprecation path and migration note. Refocused modules remain importable during
the 0.x series but may receive narrower contracts; any removal or behavior break
requires a deprecation warning in one minor release and release notes describing
the replacement. Internal implementation names may change without that promise.

New feature gate
----------------

Before adding public surface, a proposal must identify which supported use case
it advances, state its evaluator-contract effect, and show why it belongs to
lc0 interoperability or chess-decision evidence. If it instead owns a generic
interpretability technique, it belongs in an external integration or example.
