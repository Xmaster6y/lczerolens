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

            Interpretability for lc0 networks

        **lczerolens** is a package for interpreting and manipulating the neural networks produce by lc0

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

          Load a network from lc0 (``.pb`` or ``.onnx``) and load it with lczerolens using ``torch``.

    .. grid-item::

      .. div:: features-container

        .. image:: _static/images/two.png
          :width: 150

        .. div::

          **Interpretability**

          Easily compute saliency maps or aggregated statistics using the pre-built Interpretability methods.
