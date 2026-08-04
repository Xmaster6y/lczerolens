Getting Started
===============

``lczerolens`` makes lc0-family models portable and operable in PyTorch, then
expresses evaluator and search behaviour as chess-domain evidence. It owns the
model and chess-analysis boundary, not a neural-interpretability method.
External tools such as ``tdhook``, ``captum``, ``zennit``, and ``nnsight`` can
instrument that boundary without becoming lczerolens dependencies.

Installation
------------

.. code-block:: console

   pip install lczerolens

Use the ``hub`` extra for Hugging Face model loading, ``backends`` for optional
lc0 bindings, ``viz`` for rendering helpers, and ``datasets`` for dataset
adapters and concept metrics.

Supported user path
-------------------

* :doc:`scope` defines the board, evaluator, and compatibility contract.
* :doc:`facts` covers exact facts, move/variation evidence, and constrained
  counterfactual positions.
* :doc:`search` defines typed, capability-aware traces. ``ReferenceMCTS`` is
  an auditable reference implementation, not production lc0 search.
* :doc:`behavior` covers evaluator, counterfactual, and search comparisons.
* :doc:`api/index` contains generated reference documentation for every
  importable module.

Notebook status
---------------

The repository retains older notebooks as historical examples, but they are
not a rendered or supported tutorial path. Their optional dependencies, runtime
assumptions, and completion status vary; use the maintained pages above for
current behaviour and compatibility commitments.
