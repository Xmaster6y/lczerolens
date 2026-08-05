Features
========

The maintained documentation describes the supported library surface rather
than treating notebooks as API guarantees.

* :doc:`scope` defines board encoding, evaluator outputs, model formats, and
  the external-integration boundary.
* :doc:`facts` covers exact position facts, move and line analysis, and
  constrained counterfactuals.
* :doc:`search` defines provenance, capabilities, snapshots, and the
  deterministic reference-search boundary.
* :doc:`behavior` defines evaluator, counterfactual, and search comparisons.

Historical notebook examples remain in the source repository but are not
published as finished tutorials or compatibility commitments.

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/features/encode-boards.ipynb
   notebooks/features/load-models.ipynb
   notebooks/features/move-prediction.ipynb
   notebooks/features/run-models-on-gpu.ipynb
   notebooks/features/evaluate-models-on-puzzles.ipynb
   notebooks/features/convert-official-weights.ipynb
   notebooks/features/visualise-heatmaps.ipynb
   notebooks/features/probe-concepts.ipynb
   notebooks/features/selfplay-mcts-nn.ipynb
