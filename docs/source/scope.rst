Scope and compatibility
=======================

``lczerolens`` makes lc0-family models operable in PyTorch and turns evaluator
and search behavior into chess-domain evidence.

Supported
---------

The public workflows are demonstrated in :doc:`tutorials`:

* load ONNX, serialized PyTorch, or Hugging Face models;
* encode and evaluate one or many ``chess.Board`` positions;
* inspect legal policy, WDL, value, and MLH outputs;
* analyze exact facts, moves, and variations;
* define and grade authored puzzles;
* construct validity-aware counterfactual positions;
* run reference search or translate public lc0 output;
* compare evaluator and search decisions; and
* freeze and serialize reproducible evidence.

Model compatibility
-------------------

``LczeroModel.from_path`` accepts converted ``.onnx`` networks and serialized
``.pt`` modules. ``LczeroModel.from_hf`` downloads through the optional ``hub``
extra. Arbitrary PyTorch modules can be wrapped by declaring their output
heads.

Official lc0 :cite:p:`lczero` weights are converted externally. Native bindings
are a conformance-test dependency, not a runtime abstraction.

Ownership boundary
------------------

The project owns lc0/PyTorch interoperability, chess-aware evaluator semantics,
typed search evidence, exact chess analysis, and reproducible comparison
records.

It does not own chess rules, production-engine behavior, generic datasets,
hooks, attribution, probing, sparse autoencoders, natural-language coaching,
or scientific conclusions. Those systems integrate through the model,
TensorDict, evaluator, or evidence boundaries.

Compatibility
-------------

The supported surface is the documented package API and executable notebook
workflows. Private codec details and unexported helpers may change. New public
features must advance a supported workflow and belong to lc0 interoperability
or chess-decision evidence.
