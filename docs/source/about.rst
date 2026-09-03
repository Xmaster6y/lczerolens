About lczerolens
================

``lczerolens`` provides PyTorch interoperability and chess-analysis evidence
for lc0-family networks. Start with :doc:`start`, learn from the executable
:doc:`tutorials`, and use :doc:`api/index` when you need exact signatures.

Related projects
----------------

* `Leela Chess Zero <https://lczero.org/>`_ :cite:p:`lczero`
* `python-chess <https://python-chess.readthedocs.io/>`_ :cite:p:`python-chess`
* `TensorDict <https://docs.pytorch.org/tensordict/>`_ :cite:p:`tensordict`

Interpretability methods are external integrations rather than core
lczerolens APIs.

Research context
----------------

lc0-family networks sit in the policy-and-value search lineage of AlphaZero
:cite:p:`silver2018`. ``ReferenceSearch`` is a transparent reference tool, not
a reproduction of an engine; its tree-search vocabulary is contextualized by
UCT :cite:p:`kocsis2006`. The Maia model used in the getting-started example is
a human-move-prediction model :cite:p:`mcilroy-young2020`.

Citation
--------

Use the versioned :download:`CITATION.cff <../../CITATION.cff>` metadata when
you publish research with lczerolens.

.. code-block:: bibtex

   @software{poupart_lczerolens_2026,
     author = {Poupart, Yoann},
     title = {LczeroLens},
     version = {0.5.1},
     year = {2026},
     url = {https://github.com/Xmaster6y/lczerolens}
   }

``lczerolens`` is available under the MIT License.

References
----------

.. bibliography::
   :all:
