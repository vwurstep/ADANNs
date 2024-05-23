#!/bin/bash

# Execute the notebook using papermill
papermill ADANN_semilinear_heat.ipynb Z_output_ADANN_semilinear_heat_1d.ipynb \
    -p dim 1 \
    -p test_run False

papermill ADANN_semilinear_heat.ipynb Z_output_ADANN_semilinear_heat_2d.ipynb \
    -p dim 2 \
    -p test_run False   

papermill ADANN_Burgers.ipynb Z_output_ADANN_Burgers.ipynb \
    -p test_run False

papermill ADANN_Reaction_Diffusion.ipynb Z_output_ADANN_Reaction_Diffusion.ipynb \
    -p test_run False

papermill ADANN_learning_rate_experiments.ipynb Z_output_ADANN_learning_rate_experiments.ipynb \
    -p test_run False