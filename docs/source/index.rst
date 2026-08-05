:html_theme.sidebar_secondary.remove: true
:sd_hide_title:

lczerolens
==========

.. toctree::
    :maxdepth: 1
    :hidden:

    start
    features
    tutorials
    facts
    search
    use-cases
    scope
    architecture
    api/index
    About <about>

.. grid:: 1 1 2 2
    :class-container: hero
    :reverse:

    .. grid-item::
        .. div::

          .. image:: _static/images/lczerolens-logo.svg
            :width: 300
            :height: 300

    .. grid-item::

        .. div:: sd-fs-1 sd-font-weight-bold title-bot sd-text-primary image-container

            LczeroLens

        .. div:: sd-fs-4 sd-font-weight-bold sd-my-0 sub-bot image-container

            lc0 interoperability and chess-decision analysis

        PyTorch evaluation and chess-decision evidence for lc0-family models.

        .. div:: button-group

          .. button-ref:: start
            :color: primary
            :shadow:

                  Get Started

          .. button-ref:: use-cases
            :color: primary
            :outline:

                Use Cases

          .. button-ref:: api/index
            :color: primary
            :outline:

                API Reference


.. div:: sd-fs-1 sd-font-weight-bold sd-text-center sd-text-primary sd-mb-5

  Find what you need

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 150

        .. div::

          **Examples**

          Start with model loading, position evaluation, chess evidence, or replayable search.

          :doc:`Open examples <tutorials>`

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **Release use cases**

          See supported workflows and the guarantees that make their outputs usable.

          :doc:`Read use cases <use-cases>`

    .. grid-item::

      .. div:: features-container

        .. div::

          **Architecture**

          See the chess, model, evaluator, and evidence boundaries.

          :doc:`Read architecture <architecture>`

    .. grid-item::

      .. div:: features-container

        .. div::

          **API reference**

          Browse the supported package surface.

          :doc:`Open API reference <api/index>`
