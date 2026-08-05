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

        **lczerolens** makes lc0-family models operable in PyTorch and turns their evaluator and search behavior into chess-domain evidence.
        Interpretability frameworks such as `tdhook`, `captum`, `zennit`, and `nnsight` integrate at this boundary; they are not core abstractions.

        .. div:: button-group

          .. button-ref:: start
            :color: primary
            :shadow:

                  Get Started

          .. button-ref:: tutorials
            :color: primary
            :outline:

                Tutorials

          .. button-ref:: api/index
            :color: primary
            :outline:

                API Reference


.. div:: sd-fs-1 sd-font-weight-bold sd-text-center sd-text-primary sd-mb-5

  Key Features

.. grid:: 1 1 2 2
    :class-container: features

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/one.png
          :width: 150

        .. div::

          **Adaptability**

          Load converted Lczero ONNX or serialized PyTorch networks through one TensorDict model boundary.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **Composable execution**

          Run board-aware evaluation while external tools instrument the same TensorDict execution path.
