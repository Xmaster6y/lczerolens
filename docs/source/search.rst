Search ownership and trace contract
===================================

Status
------

This page is the architecture decision for the search surface introduced by
issue #131. It fixes the contract that the deterministic reference search and
the lc0 adapter must implement; those implementations belong to follow-up
issues #140 and #141.

Decision and ownership
----------------------

Production-strength search belongs to lc0 or another chess engine.
``lczerolens`` may launch or attach to such an engine and translate evidence it
actually exposes, but it does not own engine strength, time management, tree
reuse, batching, collision handling, pruning, transpositions, or lc0 fidelity.

``lczerolens`` does own an engine-independent trace vocabulary and a small,
deterministic neural-MCTS reference implementation. The reference search exists
for auditability, replay, hermetic tests, and research instrumentation. It is
not a production engine and must not be presented as numerically or
behaviorally equivalent to lc0.

Public schema
-------------

The typed records live in :mod:`lczerolens.search_trace`. A
:class:`~lczerolens.search_trace.SearchTrace` contains root position identity,
producer provenance, one or more root snapshots, an advertised capability, and
optional per-simulation events. ``None`` means a field was unavailable from the
source. Adapters must not calculate a plausible value merely to fill an absent
field.

The schema uses stable UCI move strings at adapter boundaries. FEN identifies
the root position, including side to move and clocks. Provenance records source,
engine/version, network identity/checksum when available, and source-specific
parameters. A :class:`~lczerolens.search_trace.SearchBudget` distinguishes the
requested budget from the observed budget and states its unit. Node, visit,
simulation, and depth budgets are non-negative integers; time budgets may be
fractional milliseconds.

Root snapshots contain an optional position evaluation, selected action, and
root-action records. A principal variation is only the move sequence reported
by the producer; an adapter must not extend it. Reference-search events contain
the selected node/edge path, leaf record, optional expansion, backups, and
pre/post root edge state. Stable string node and event identifiers are assigned
by the producer. ``full_events`` means the emitted simulation events are
available; it does not mean every private engine node or a complete final tree
was exported.

Statistic and value semantics
-----------------------------

Every scalar evaluation, WDL record, and edge-statistics record names its
:class:`~lczerolens.search_trace.ValuePerspective`. ``root_player`` means the
side to move in the trace's root FEN; ``side_to_move`` means the side to move at
the position where the record occurs. ``white`` and ``black`` are absolute.
Comparisons and backups must first put values in the same perspective. The
reference search alternates the sign once for every parent/child ply. A terminal
win, draw, or loss is valued ``+1``, ``0``, or ``-1`` for the stated
perspective.

WDL is ordered ``(win, draw, loss)`` for its stated perspective, contains
probabilities that sum to one, and converts to a scalar as
``win + draw_score * draw - loss``. The default draw score is zero. lc0's
reported ``WL = win - loss`` and ``D = draw`` convert exactly as
``win = (1 - D + WL) / 2`` and ``loss = (1 - D - WL) / 2``. A source may provide
scalar value, WDL, or both; one is not silently derived from the other.

For an edge from a node to a legal action:

* ``P`` (``prior``) is the normalized probability assigned to that action at
  expansion. It is non-negative; when every exposed root action has ``P``,
  those priors sum to one.
* ``N`` (``visits``) is the number of completed traversals backed up through
  that edge at the time of the record.
* ``W`` (``total_value``) is the sum of backed-up scalar values in the edge's
  stated perspective.
* ``Q`` (``mean_value``) is ``W / N``. Its convention is exactly zero when
  ``N == 0``.
* ``U`` (``exploration``) is the exploration contribution reported by the
  producer at selection time. Its formula is source-specific, so adapters
  retain it only when exposed and never recompute it as if all engines used the
  reference PUCT formula.

A leaf evaluation is the value first available at the selected leaf, before
sign-changing backups. Reference events additionally record evaluator dtype,
the evaluator and search devices, and legal policy logits when an evaluator was
called. A :class:`~lczerolens.search_trace.BackupUpdate` records the signed value
used at that edge plus its pre/post statistics. The reference implementation in
#140 must enforce ``N_post = N_pre + 1``, ``W_post = W_pre + signed_value``, and
``Q_post = W_post / N_post``.

The selected root action is recorded separately from its statistics. Its rule,
temperature when applicable, and tie-break are mandatory. The deterministic
reference search uses maximum visit count at temperature zero and stable UCI
lexicographic order for exact ties. Engine adapters state ``engine-defined``
when the engine does not expose a more precise tie rule.

Capabilities
------------

Capabilities are ordered promises:

``root_result``
  A root evaluation and/or selected move is available. Action details may be
  absent. A trace must contain at least one snapshot with one of these root
  fields, though an intermediate snapshot may contain only action statistics.

``root_action_stats``
  The producer exposed the legal root action collection and available P/N/W/Q/U
  fields. Individual statistics remain ``None`` if the source omitted them.

``root_snapshots``
  Budget-labelled root action statistics are available at one or more points.

``full_events``
  Append-only per-simulation path, leaf, expansion, and backup event records are
  available. This is still not a claim of private full-tree access.

``replayable``
  Every event also contains the backup and pre/post root state required for an
  independent replayer to reconstruct the emitted root statistics. Each state
  is non-empty, contains exactly one edge transition matching a recorded
  backup, chains from the preceding event, and the final state equals the final
  root snapshot.

Consumers call :meth:`~lczerolens.search_trace.SearchTrace.supports` or
:meth:`~lczerolens.search_trace.SearchTrace.require` before making a claim.
Requesting full events or replay from an lc0 root-only trace raises
:class:`~lczerolens.search_trace.SearchCapabilityError`. Schema construction
also rejects a capability label when its required records are absent.

Representative mappings
-----------------------

The current Python ``MCTS`` can be represented at
``root_action_stats``: its legal moves map to UCI strings, policy to ``P``,
visits to ``N``, and ``q_values`` to ``Q``. It does not retain cumulative
``W``, source ``U``, budget snapshots, or append-only events, so those fields
remain absent. Tests exercise this record without upgrading its capability.

Captured lc0 ``VerboseMoveStats`` or ``LogLiveStats`` output maps its public
``info string <move>`` records and available P/N/Q/U/PV fields into a
budget-labelled root snapshot. ``M``, ``S``, and ``Q+U`` are recognised but
have no counterpart in the shared schema; ``W`` is recorded only when the
source emits it. Reported ``WL`` and ``D`` form the action's search evaluation,
while ``V`` is its separately labelled leaf/network evaluation; unavailable
entries remain absent. Display-rounded P values are normalised across a fully
exposed root action set before the schema's unit-sum invariant is checked. UCI
``bestmove`` supplies the legal selection; terminal root FENs are rejected
before starting the engine. Version, network checksum, command options, and
requested/observed nodes or time belong to provenance and budget records.
Unless lc0 exposes simulation transitions through a separately supported
interface, the adapter advertises at most ``root_snapshots``. Tests exercise a
representative captured-output-shaped record without an lc0 binary. The optional
:class:`~lczerolens.lc0_adapter.Lc0RootSnapshotParser` implements this
versioned captured-output contract; its paired process adapter invokes a
user-supplied UCI binary. Live conformance is opt-in through
``LC0_EXECUTABLE``, ``LC0_NETWORK``, and ``LC0_VERSION``.

Adapter boundary
----------------

Search producers implement outside the schema records. The reference adapter
turns deterministic state transitions into ``SimulationEvent`` values. The lc0
process adapter owns command invocation and version-specific parsing, then
returns the same ``SearchTrace`` type. Other engine adapters may do likewise.
Parsers must fail with actionable version or shape diagnostics rather than
silently discard a field they claim to support.

Existing API disposition
------------------------

The existing search implementation remains importable through the 0.x
compatibility window while #140 supplies its replacement contract:

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Symbol or option
     - Decision
     - Rationale
   * - ``MCTS``
     - Retain temporarily
     - The mutable legacy API stays importable through 0.x; new auditable search uses ``ReferenceMCTS``.
   * - ``Node``
     - Internalize
     - Mutable tree storage is an implementation detail; trace records are the public evidence boundary.
   * - ``Heuristic`` and ``ModelHeuristic``
     - Refocus
     - Adapt the stable evaluator contract for reference search rather than define a general game-search API.
   * - ``RandomHeuristic`` and ``MaterialHeuristic``
     - Retain for tests
     - Deterministic/reference fixtures only; they are not evaluator standards.
   * - ``MCTSSampler``
     - Refocus
     - Remains a compatibility move-selection facade over the reference search, not an engine service.
   * - ``MCTSSampler.use_q_values``
     - Deprecate
     - Root choice becomes an explicit selection policy over typed action statistics.
   * - ``MCTS.n_parallel_rollouts``
     - Deprecate
     - The v1 reference search is sequential and deterministic; parallel production search stays engine-owned.
   * - ``MCTS.render_tree``
     - Internalize
     - Rendering mutable implementation nodes is not the portable trace contract.

Issue #140 implements the new reference API without breaking the existing
mutable API. A future major-version migration may rename or remove the legacy
surface after downstream callers have moved to typed traces.

Reference implementation
------------------------

:class:`~lczerolens.reference_search.ReferenceMCTS` is the #140 implementation.
It consumes the #130 single-board evaluator TensorDict (a raw 1858-logit
``policy`` and scalar side-to-move ``value``) and returns a
:class:`~lczerolens.search_trace.SearchTrace` with ``replayable`` capability.
The public :func:`~lczerolens.reference_search.replay_root_events` helper
reconstructs final root statistics using events alone; it neither reads mutable
tree state nor calls the search backup routine.

The reference PUCT score is ``Q + c_puct * P * sqrt(sum(N)) / (1 + N)``.
Exact score ties and zero-visit root selection use UCI lexicographic order;
final root selection is maximum visit count at temperature zero. Search is
strictly sequential. It intentionally omits lc0 FPU, batching, virtual visits,
collisions, pruning, transpositions, noise, and tree reuse, and therefore
makes no lc0-equivalence or playing-strength claim.

The legacy ``MCTS``/``Node`` API remains available during the 0.x compatibility
window. New audit or research code should use ``ReferenceMCTS.search`` and its
typed trace rather than depending on the mutable legacy tree.
