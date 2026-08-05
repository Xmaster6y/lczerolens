Tutorials
=========

All tutorials use deterministic fixtures and run offline during the docs build.

Feature notebooks
-----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Models and inputs
      :link: notebooks/features/models-and-inputs
      :link-type: doc
      :class-card: surface

      :octicon:`cpu;2em;sd-text-primary`

      Prepare boards, input planes, legal masks, devices, and hooks.

   .. grid-item-card:: Evaluate positions
      :link: notebooks/features/evaluate-positions
      :link-type: doc
      :class-card: surface

      :octicon:`pulse;2em;sd-text-primary`

      Work with legal policies, heads, batches, and evaluation records.

   .. grid-item-card:: Chess evidence
      :link: notebooks/features/chess-evidence
      :link-type: doc
      :class-card: surface

      :octicon:`telescope;2em;sd-text-primary`

      Keep rule-exact analysis and authored correctness distinct from preference.

   .. grid-item-card:: Replayable search
      :link: notebooks/features/replayable-search
      :link-type: doc
      :class-card: surface

      :octicon:`history;2em;sd-text-primary`

      Run deterministic search, inspect capabilities, and replay trace evidence.

Tutorial notebooks
------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Decision analysis
      :link: notebooks/tutorials/decision-analysis
      :link-type: doc
      :class-card: surface

      :octicon:`workflow;2em;sd-text-primary`

      Compose evaluation, search, exact chess evidence, and persistence.

   .. grid-item-card:: Compare models
      :link: notebooks/tutorials/compare-models
      :link-type: doc
      :class-card: surface

      :octicon:`git-compare;2em;sd-text-primary`

      Compare pinned evaluator outputs on the same positions.

   .. grid-item-card:: Analyze puzzles
      :link: notebooks/tutorials/analyze-puzzles
      :link-type: doc
      :class-card: surface

      :octicon:`tasklist;2em;sd-text-primary`

      Grade authored solution trees and inspect accepted lines.

.. toctree::
   :hidden:
   :maxdepth: 2

   decision-analysis-tutorial
   notebooks/features/models-and-inputs
   notebooks/features/evaluate-positions
   notebooks/features/chess-evidence
   notebooks/features/replayable-search
   notebooks/tutorials/decision-analysis
   notebooks/tutorials/compare-models
   notebooks/tutorials/analyze-puzzles
