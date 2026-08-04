End-to-end decision analysis
============================

This maintained tutorial validates the complete product boundary with one
reproducible decision-analysis workflow. It intentionally uses a tiny,
versioned PyTorch fixture rather than a downloaded network: its observations
are demonstrations of API composition, **not scientific conclusions about
chess or a trained model**.

Run it from a clean environment with the development group installed::

   uv sync --group dev
   uv run pytest -q -m integration tests/integration/test_decision_analysis_tutorial.py

The test is an explicit integration tier, so normal unit runs do not silently
present this tutorial as model validation. The fixture is defined alongside the
tutorial and has only the standard ``policy`` and ``value`` TensorDict heads.

The workflow
------------

The executable companion is :download:`examples/decision_analysis_tutorial.py
<../../examples/decision_analysis_tutorial.py>`. Its ``run_tutorial`` function:

#. wraps a small PyTorch module in :class:`~lczerolens.model.LczeroModel`, the
   same TensorDict evaluator boundary used for an lc0-family model loaded from
   ONNX, Torch, or the Hub;
#. evaluates the initial position after masking to legal candidates;
#. runs :class:`~lczerolens.reference_search.ReferenceMCTS` and retains its
   capability-aware :class:`~lczerolens.search_trace.SearchTrace`;
#. attaches exact material variation evidence to the evaluator and search
   candidates;
#. makes a constrained sibling-move counterfactual (``g1f3`` versus ``b1c3``);
#. reports the target ``e7e5`` effect separately from the collateral legal
   action distribution; and
#. records whether evaluator and reference-search candidates agree, including
   their root statistics and variation evidence.

The reference search is explicitly ``replayable`` and therefore exposes full
events. That is stronger than an official-lc0 adapter that only advertises
root snapshots. Do not use this tutorial to claim that the two searches are
algorithmically equivalent, or that root-only output contains event evidence.

Optional external instrumentation
---------------------------------

An attribution or hook library belongs *around* ``model`` in
``load_fixture_evaluator``/``evaluate``. For example, an application can wrap
the returned ``LczeroModel`` with a tdhook or Captum session, then preserve the
same ``TensorDict`` output for ``evaluator_behavior`` and ``ReferenceMCTS``.
Removing that optional wrapper changes neither the chess-evidence nor the
search-analysis APIs. The maintained executable deliberately does not depend
on an interpretability package, so the product boundary remains reproducible
with the documented development environment.

Reading the result
------------------

``TutorialResult`` retains structured records instead of prose explanations:
the raw evaluator preference, the search decision comparison, exact variation
evidence, the counterfactual validity tier, one target-move delta, and the
collateral distribution shift. Those are observations with provenance. They
can motivate a research question, but they do not establish a causal mechanism,
model strength, or general chess conclusion.
