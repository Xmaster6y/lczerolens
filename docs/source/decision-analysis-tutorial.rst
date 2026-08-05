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

The release boundary runs the same workflow from the built wheel in a fresh
virtual environment, with isolated Python startup and an explicit assertion
that ``lczerolens.__file__`` is outside the checkout::

   just tests-wheel

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
#. runs :class:`~lczerolens.search.reference.ReferenceSearch` and retains its
   natural :class:`~lczerolens.search.result.SearchResult` plus detailed trace;
#. attaches exact material line analysis to the evaluator and search
   candidates;
#. makes a constrained sibling-move counterfactual (``g1f3`` versus ``b1c3``);
#. constructs and solves an authored mate-in-one puzzle independently of model
   or search preference;
#. reports the target ``e7e5`` effect separately from the collateral legal
   action distribution; and
#. records whether evaluator and reference-search candidates agree, including
   their root statistics and line analysis;
#. serializes the complete :class:`~lczerolens.decision.DecisionAnalysis` as
   canonical, versioned JSON; and
#. restores the artifact and verifies equality plus the versioned fixture's
   stable SHA-256 digest.

Persisting the result
---------------------

Pass an output path to retain the exact artifact exercised by the tutorial::

   from examples.decision_analysis_tutorial import run_tutorial

   result = run_tutorial("decision.json")
   assert result.restored_decision == result.decision
   assert result.restored_decision.digest() == result.decision_digest

The artifact composes existing canonical evaluator and search-trace records
with the exact line and counterfactual evidence retained by the decision.
Loading reconstructs the natural :class:`~lczerolens.search.result.SearchResult`
and revalidates position identity, selected actions, producer provenance, and
every domain record. It never stores the model, evaluator, tensors, devices,
hooks, or engine processes.

The reference search is explicitly ``replayable`` and therefore exposes full
events. That is stronger than an official-lc0 adapter that only advertises
root snapshots. Do not use this tutorial to claim that the two searches are
algorithmically equivalent, or that root-only output contains event evidence.

Optional external instrumentation
---------------------------------

An attribution or hook library belongs *around* ``model`` in
``load_fixture_evaluator``/``evaluate``. For example, an application can wrap
the returned ``LczeroModel`` with a tdhook or Captum session, then preserve the
same ``TensorDict`` output for evaluator analysis and ``ReferenceSearch``.
Removing that optional wrapper changes neither the chess-evidence nor the
search-analysis APIs. The maintained executable deliberately does not depend
on an interpretability package, so the product boundary remains reproducible
with the documented development environment.

Reading the result
------------------

``TutorialResult`` retains structured records instead of prose explanations:
the raw evaluator preference, the search decision comparison, exact variation
evidence, an authored puzzle attempt, the counterfactual validity tier, one
target-move delta, and the collateral distribution shift. It also retains the
restored decision and its canonical digest. Those are observations with
provenance. They can motivate a research question, but they do not establish a
causal mechanism, model strength, or general chess conclusion.
