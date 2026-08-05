Chess facts and evidence
========================

Run :doc:`notebooks/features/chess-evidence` for the complete feature path or
:doc:`notebooks/tutorials/analyze-puzzles` for a composed workflow.

Facts
-----

The bundled analyzers report exact ``python-chess`` observations:

* material and piece presence;
* attackers and defenders;
* check status; and
* legal mobility.

Each ``Evidence`` record retains its subject, perspective, guarantee,
provenance, and history requirement. ``EvidenceSet`` composes records without
discarding those distinctions. Evaluator or search preferences are
model-derived evidence, not exact position facts.

Moves and lines
---------------

``analyze_move`` records the position before and after a legal move, concrete
pieces, special-move effects, and changed or preserved facts. ``analyze_line``
does the same for every ply and records history and terminal status.

These APIs describe rule-exact changes. They do not infer initiative,
compensation, or another strategic interpretation. A claimable draw is kept
separate from a forced terminal result. Invalid lines fail with a structured
``LineAnalysisError``.

Counterfactuals and puzzles
---------------------------

``sibling_counterfactual`` compares two legal children of the same parent.
Structural removal or relocation can establish rule validity, but not
historical reachability. Every result states the strongest validity guarantee
it actually established.

A ``Puzzle`` is an authored task. Its solution tree—not evaluator preference,
search preference, or terminality—defines accepted moves and whether an
attempt is solved.
