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
lc0 bindings, and ``viz`` for rendering helpers.

Supported user path
-------------------

* :doc:`use-cases` defines the concrete workflows that gate the breaking
  ``0.5`` refactor and release.
* :doc:`architecture` defines the target TensorDict execution contract,
  chess-facing objects, ownership boundaries, and deletion policy.
* :doc:`scope` documents the currently implemented pre-refactor surface; the
  architecture page supersedes its compatibility policy for the ``0.5`` work.
* :doc:`facts` covers exact facts, move/line analysis, and constrained
  counterfactual positions.
* :doc:`search` defines the shared result, typed limits, and detailed traces.
  ``ReferenceSearch`` is an auditable oracle, not a production engine.
* :doc:`behavior` covers evaluator, counterfactual, and search comparisons.
* :doc:`decision-analysis-tutorial` runs those public boundaries together with
  a small versioned fixture; its output is a demonstration, not a result.
* :doc:`api/index` contains generated reference documentation for every
  importable module.

The maintained tutorial is executable and tested. Historical notebooks built
on removed samplers, datasets, concepts, and mutable search objects are not part
of the release surface.
