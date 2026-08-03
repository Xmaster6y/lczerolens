Chess facts and evidence
========================

``lczerolens.facts`` represents chess observations before a consumer turns them
into labels, tensors, datasets, metrics, or natural-language claims. Every
``Evidence`` record states its kind, scope, subject, value or undefined reason,
perspective, guarantee, supporting chess objects, history requirement, and
versioned analyzer provenance.

Reference analyzers
-------------------

The bundled analyzers use only ``python-chess`` and operate on one position:

* ``MaterialAnalyzer`` records a side's material points and every counted piece;
* ``PiecePresenceAnalyzer`` records whether a side has a piece type and where;
* ``AttacksDefendersAnalyzer`` records attackers and defenders of a square;
* ``CheckStatusAnalyzer`` records the king square and checking pieces; and
* ``LegalMobilityAnalyzer`` records the count and complete set of legal moves.

These analyzers carry the ``exact`` guarantee. An invalid position or missing
king is returned as explicit undefined evidence where it prevents the requested
semantics. ``history_is_available`` distinguishes boards played forward with a
move stack rooted at the standard initial position from positions reconstructed
from an analysis FEN, including FENs with reset move counters.

Composition and guarantees
--------------------------

``EvidenceSet.compose`` and ``EvidenceSet.filter`` retain the original evidence
records. Exact, heuristic, engine-derived, and search-derived observations can
therefore coexist without losing their tags. Extracting bare values through
``EvidenceSet.values`` requires the caller to name one guarantee and rejects a
mixed set.

Migration from concepts
-----------------------

The existing ``lczerolens.concepts`` API remains available during the 0.x
compatibility period. It computes task-ready labels and optionally owns dataset
feature or metric conversion. New chess-semantic code should instead compute an
``Evidence`` first, then explicitly convert ``evidence.value`` for its task.

The direct migrations are:

* ``HasPiece`` to ``PiecePresenceAnalyzer``;
* ``HasMaterialAdvantage`` to two ``MaterialAnalyzer`` results followed by an
  explicit comparison; and
* rule-based threat labels to ``AttacksDefendersAnalyzer`` plus an explicit
  task predicate.

``BestLegalMove`` and ``PieceBestLegalMove`` are model-derived labels, not exact
position facts. Keep using them for compatibility or represent a new analyzer's
result with ``engine-derived`` or ``search-derived`` guarantee as appropriate.
No dataset, scikit-learn, neural hook, or attribution dependency is imported by
the core fact API.
