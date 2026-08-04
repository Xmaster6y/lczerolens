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

Move deltas and variation evidence
----------------------------------

``lczerolens.move_evidence`` composes the exact fact analyzers across legal
moves. ``analyze_move_delta`` records the before and after FEN, concrete mover,
moving and captured pieces, special-move metadata, and the original evidence on
both sides of every changed fact. Its ``created`` and ``removed`` collections
are set-difference views, while ``preserved`` retains unchanged controls and
``transitions`` links corresponding before/after records.

Calling ``analyze_move_delta`` without an explicit analyzer list uses
``exact_move_analyzers``: material, check status, piece presence, and
attack/defender facts for both concrete sides, plus side-to-move mobility. This
makes alternating perspectives explicit instead of silently comparing "us" at
one ply with a different "us" at the next.

``analyze_variation`` applies the same comparison to every ply of an ordered
legal line. Its position snapshots state whether the board carries complete
standard-game history. Truncated histories can be retained explicitly or
rejected with ``HistoryPolicy.REQUIRE_COMPLETE``. Empty or illegal lines,
history incompatibility, and declared candidate/response mismatches raise
``VariationAnalysisError`` with a stable ``VariationFailureReason`` and
position/ply context.

``VariationIntent`` can mark a line as neutral, as support for a named claim by
a candidate, or as an opponent response that refutes that claim. These records
do not generate prose and do not infer heuristic notions such as initiative or
compensation. Terminal result metadata uses rule-exact ``python-chess``
outcomes. ``VariationTerminal.claimable_draw`` separately records a threefold
or fifty-move draw that can be claimed when complete history is available; it
does not turn a legally continuable position into a terminal result.
