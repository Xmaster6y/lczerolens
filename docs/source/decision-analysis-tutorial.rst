End-to-end decision analysis
============================

Use :doc:`notebooks/tutorials/decision-analysis` for the interactive workflow.
The downloadable :download:`Python companion
<../../examples/decision_analysis_tutorial.py>` runs the same versioned,
offline fixture from a script.

Run it
------

.. code-block:: console

   uv sync --group dev
   uv run pytest -q -m integration tests/integration/test_decision_analysis_tutorial.py

``just tests-wheel`` additionally runs the maintained workflow from an
installed wheel in an isolated environment.

What it composes
----------------

The workflow:

* evaluates a position and runs replayable reference search;
* attaches exact line and counterfactual evidence;
* grades an authored puzzle independently of model preference;
* compares policy and search candidates; and
* serializes, restores, and digest-checks the complete decision record.

Pass a path to ``run_tutorial`` to retain the canonical JSON artifact. The
artifact stores immutable evidence, not models, tensors, devices, hooks, or
engine processes.

The fixture demonstrates API composition only. It is not a trained chess
model, a strength result, or evidence of a causal mechanism. Reference-search
events also do not imply that official lc0 root output is replayable.
