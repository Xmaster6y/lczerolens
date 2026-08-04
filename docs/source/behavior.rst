Observable behaviour comparisons
================================

``lczerolens.behavior`` compares what an evaluator or search source actually
exposed. It is an evidence boundary: it does not implement attribution, hooks,
probes, sparse autoencoders, or natural-language explanation algorithms.

Evaluator behaviour
-------------------

:func:`~lczerolens.behavior.evaluator_behavior` accepts one
:class:`~lczerolens.board.LczeroBoard` and the canonical evaluator output.
``policy`` is required and interpreted as 1858 raw logits. Illegal actions are
removed before a legal-action softmax. Candidate ranks use competition ranking;
exact maximum-logit ties are retained and the selected representative uses UCI
lexicographic order. Value, WDL, and MLH are optional and remain absent instead
of being derived from another head. Values and WDL state their perspective.

All definitions are available through
:data:`lczerolens.behavior.METRIC_DEFINITIONS`. Each definition fixes
perspective, normalization, single-position or snapshot aggregation, ties,
missing/illegal-action handling, and any required search capability.

Search behaviour
----------------

:func:`~lczerolens.behavior.compare_evaluator_to_search` requires root action
statistics and returns evaluator probability/rank alongside P, N, visit share,
Q, PV, visit amplification, and final selection for each exposed root action.
Policy-to-search divergence is total variation between visit shares and
evaluator probabilities conditional on the exposed action set. The original
evaluator probability mass covered by that set is reported separately, so a
partial public engine record cannot be mistaken for a complete legal-action
comparison.

Budget-labelled traces additionally expose candidate-rank and Q evolution,
selected-move changes, the first budget at which a move received a visit, and
adjacent-PV prefix stability. Unavailable source fields remain ``None``.

Event metrics are deliberately separate. Calling
:func:`~lczerolens.behavior.compare_search_events` on a root-only trace raises
:class:`~lczerolens.search_trace.SearchCapabilityError`. Path-depth and
expansion summaries require ``full_events``; independent root transition
validation additionally requires ``replayable``. Official lc0 public root
output and :class:`~lczerolens.reference_search.ReferenceMCTS` may therefore be
compared through the same root vocabulary without claiming that their search
algorithms, strengths, or capabilities are equivalent.

Counterfactual behaviour and controls
-------------------------------------

:func:`~lczerolens.behavior.compare_counterfactual_behavior` takes two
standardized evaluator records and a non-empty set of target moves. Every
target receives its own probability, rank, and selection effect. All other
legal actions form a separate collateral record, including union-set total
variation when the legal move sets differ. ``matched``, ``shuffled``, and
``wrong_target`` are first-class :class:`~lczerolens.behavior.ControlKind`
values rather than labels hidden in prose.

A successful :class:`~lczerolens.counterfactuals.CounterfactualResult` and
supplied :class:`~lczerolens.move_evidence.VariationEvidence` can be retained
with the comparison. :func:`~lczerolens.behavior.compare_search_decision`
combines the evaluator's initial candidate, the source-selected search
candidate, their root statistics, source capability, and any supplied
variation evidence. It provides the structured checkpoint-D evidence needed
to ask why search preferred one candidate without generating or asserting a
heuristic chess explanation.
