Getting Started
===============

Installation
------------

.. code-block:: console

   pip install lczerolens

Use ``pip install "lczerolens[hub]"`` for Hugging Face model loading. The
example uses Maia, a model family trained to predict human chess moves
:cite:p:`mcilroy-young2020`. Then run an evaluation:

.. code-block:: python

   import chess
   from lczerolens import LczeroEvaluator, LczeroModel

   model = LczeroModel.from_hf("lczerolens/maia-1100")
   evaluator = LczeroEvaluator(model)
   evaluation = evaluator.evaluate(chess.Board())

   print(evaluation.policy.best_move)

Examples
--------

Start with the notebook closest to your task:

* :doc:`notebooks/features/models-and-inputs` — load a model and inspect its
  TensorDict inputs;
* :doc:`notebooks/features/evaluate-positions` — evaluate positions and
  batches;
* :doc:`notebooks/features/chess-evidence` — analyze moves, lines, puzzles,
  and counterfactuals; and
* :doc:`notebooks/features/replayable-search` — run and replay reference
  search.

For an end-to-end workflow, use :doc:`tutorials`. Project boundaries and the
generated reference are in :doc:`scope`, :doc:`architecture`, and
:doc:`api/index`.
