Tutorials
=========

The maintained user path is organised around the contract pages:

* :doc:`start` gives the evaluator and board entry point;
* :doc:`facts` explains evidence and counterfactual records;
* :doc:`search` explains trace capabilities and replay boundaries; and
* :doc:`behavior` explains observable comparisons.

The repository's notebook collection is retained as legacy or exploratory
material and is not published as a finished tutorial series. In particular,
the automated-interpretability and learned-look-ahead notebooks remain
incomplete. Neural-method examples are external integrations, not an
lczerolens-owned API or methodological guarantee.

The maintained :doc:`decision-analysis-tutorial` is the one reproducible
end-to-end product-validation workflow. It uses a small versioned fixture and
is executed in the explicit integration test tier.

.. toctree::
   :hidden:
   :maxdepth: 2

   decision-analysis-tutorial
   notebooks/walkthrough.ipynb
   notebooks/tutorials/framework-agnostic-interpretability.ipynb
   notebooks/tutorials/automated-interpretability.ipynb
   notebooks/tutorials/evidence-of-learned-look-ahead.ipynb
   notebooks/tutorials/piece-value-estimation-using-lrp.ipynb
