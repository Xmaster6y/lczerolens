Getting Started
===============

``lczerolens`` loads lc0-family models into PyTorch and turns evaluator and
search output into chess-domain evidence.

Installation
------------

.. code-block:: console

   pip install lczerolens

Use ``pip install "lczerolens[hub]"`` for Hugging Face model loading.

Learn by running
----------------

Start with the maintained notebooks:

* :doc:`notebooks/features/models-and-inputs` — load a model and inspect its
  TensorDict inputs;
* :doc:`notebooks/features/evaluate-positions` — evaluate positions and
  batches;
* :doc:`notebooks/features/chess-evidence` — analyze moves, lines, puzzles,
  and counterfactuals; and
* :doc:`notebooks/features/replayable-search` — run and replay reference
  search.

Then use :doc:`tutorials` for complete workflows. :doc:`scope` defines what
the project owns, :doc:`architecture` defines the system contract, and
:doc:`api/index` is the generated reference.
