Features
==========

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', (event) => {
      document.querySelectorAll('h5.card-title').forEach(el => {
      el.style.margin = '0';
      });
   });
   </script>

   <style>
      .toctree-wrapper {
         display: none !important;
      }
   </style>

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :link: notebooks/features/move_prediction.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-arrow-turn-up fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Move Prediction</h5>
               <p class="card-text">Make a move prediction for a given chess board.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/evaluate_models_on_puzzles.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-puzzle-piece fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Evaluate Models on Puzzles</h5>
               <p class="card-text">Evaluate a model on a set of puzzles.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/run_models_on_gpu.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-forward-fast fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Run Models on GPU</h5>
               <p class="card-text">Take advantage of GPU acceleration.</p>
            </div>
         </div>

   .. grid-item-card::
      :link: notebooks/features/convert_official_weights.ipynb
      :class-card: surface
      :class-body: surface

      .. raw:: html

         <div class="d-flex align-items-center">
            <div class="d-flex justify-content-center" style="min-width: 50px; margin-right: 20px; height: 100%;">
               <i class="fa-solid fa-arrows-rotate fa-2x"></i>
            </div>
            <div>
               <h5 class="card-title">Convert Official Weights</h5>
               <p class="card-text">Convert <strong>lc0</strong> networks to <strong>onnx</strong>.</p>
            </div>
         </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/features/evaluate_models_on_puzzles.ipynb
   notebooks/features/move_prediction.ipynb
   notebooks/features/run_models_on_gpu.ipynb
   notebooks/features/convert_official_weights.ipynb
