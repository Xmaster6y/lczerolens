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

Next steps
----------

Use :doc:`tutorials` for runnable explanations of model loading, evaluation,
chess evidence, search, comparison, and puzzle analysis. Every notebook can be
opened directly in Colab. Use the generated :doc:`api/index` when you need an
exact signature or field definition.
