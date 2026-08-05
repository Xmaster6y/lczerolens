Architecture
============

``lczerolens`` connects lc0-family neural execution to reproducible chess
evidence. The user-facing workflows live in :doc:`tutorials`; this page states
the boundaries behind them.

Execution flow
--------------

.. code-block:: text

   chess.Board
       -> LczeroEvaluator.prepare()
       -> TensorDict input
       -> LczeroModel
       -> TensorDict network output
       -> LczeroEvaluator.finish()
       -> Evaluation
       -> immutable evidence and comparisons

``chess.Board`` remains the chess position and history object. TensorDict is
the neural execution substrate. Domain records are the persistence and
analysis boundary.

Responsibilities
----------------

* ``python-chess`` owns rules, legal moves, FENs, and game history.
* ``LczeroModel`` owns loading and raw TensorDict network execution.
* ``LczeroEvaluator`` owns board encoding, legal policy, head validation, and
  standardized evaluation views.
* Facts, moves, puzzles, counterfactuals, search traces, and decisions own
  chess-domain evidence.
* External libraries own hooks, attribution, probing, interventions, and
  visualization.

TensorDict contract
-------------------

The canonical keys are:

.. code-block:: text

   input/planes
   input/legal_mask
   network/policy_logits
   network/wdl             optional
   network/value           optional
   network/mlh             optional
   evaluation/policy
   evaluation/value        native or derived from WDL

For non-terminal positions, evaluation policy is zero on illegal moves and
sums to one over legal moves. Instrumentation may add nested keys; evaluator
validation preserves them. ``LczeroKeys`` provides the public key constants.

Runtime and evidence
--------------------

Models, TensorDict batches, devices, engine processes, and mutable search state
are runtime objects. They are not serialized.

Evaluation records, exact line analysis, puzzle attempts, counterfactual
validity, search traces, and decision analyses are immutable evidence. They
retain position identity and producer provenance and use canonical versioned
serialization where supported.

Search boundary
---------------

Search accepts a board and one typed limit. ``ReferenceSearch`` provides a
deterministic replayable oracle. ``LczeroSearch`` translates only public output
from an external engine. Consumers check available evidence instead of
assuming every producer exposes root actions, snapshots, or events.

See :doc:`scope` for supported use cases and non-goals.
