Getting Started
===============

**lczerolens** makes lc0-family models portable and operable in PyTorch, then
expresses their evaluator and search behavior as chess-domain evidence. It owns
the model and chess-analysis boundary, not a neural-interpretability method.
Tools such as ``tdhook``, ``captum``, ``zennit``, and ``nnsight`` can instrument
that boundary without lczerolens depending on or abstracting over them. See
:doc:`scope` for the supported surface and compatibility policy.

.. _installation:

Installation
------------

To get started with lczerolens, install it with ``pip``.

.. code-block:: console

   pip install lczerolens

.. note::

   Core dependencies are light: mainly ``torch``, ``onnx2torch``, ``tensordict``, and ``python-chess``. Optional extras include ``matplotlib`` and ``graphviz`` (extra ``viz``) and lc0 bindings (extra ``backends``).
   Also, the Hugging Face Hub is required to load models from the Hub (extra ``hf``).

First Steps
-----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Walkthrough
      :link: notebooks/walkthrough.ipynb

      Walk through a basic usage of the package.

   .. grid-item-card:: Features
      :link: features
      :link-type: doc

      Review the basic features provided by :bdg-primary:`lczerolens`.

.. note::

   Check out the :bdg-secondary:`walkthrough` to get a better understanding of the package.

Advanced Features
-----------------

.. warning::

   This following section is under construction, not yet stable nor fully functional.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc

      See implementations of :bdg-primary:`lczerolens` through common interpretability techniques.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      See the full API reference for :bdg-primary:`lczerolens` to extend its functionality.
