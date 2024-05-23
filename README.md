# Algorithmically designed artificial neural networks (ADANNs)

This repository contains the code for the paper "Algorithmically Designed Artificial
Neural Networks (ADANNs):
Higher order deep operator learning for
parametric partial differential equations".
The paper is available on [arXiv](https://arxiv.org/abs/2302.03286).




## Running the code

All the numerical results in the paper can be reproduced by running the script `execute_notebooks.sh` in the `2_ADANNs` directory. 
More specifically, the following commands (run in the `2_ADANNs` directory) can be used to reproduce the results of specific sections of the paper:


### Section 4.1.2
```bash
papermill ADANN_semilinear_heat.ipynb Z_output_ADANN_semilinear_heat_1d.ipynb -p dim 1 -p test_run False
```


### Section 4.1.3
```bash
papermill ADANN_semilinear_heat.ipynb Z_output_ADANN_semilinear_heat_2d.ipynb -p dim 2 -p test_run False
```

### Section 4.2.2
```bash
papermill ADANN_Burgers.ipynb Z_output_ADANN_Burgers.ipynb -p test_run False
```

### Section 4.3.2
```bash
papermill ADANN_Reaction_Diffusion.ipynb Z_output_ADANN_Reaction_Diffusion.ipynb -p test_run False
```

### Section B.1
```bash
papermill ADANN_learning_rate_experiments.ipynb Z_output_ADANN_learning_rate_experiments.ipynb -p test_run False
```


## Requirements

To run the scripts and reproduce the numerical results, the following packages are needed:

- pytorch
- matplotlib
- pandas
- importlib
- openpyxl
- scipy
- seaborn
- neuraloperator
- wandb
- ruamel.yaml
- configmypy
- tensorly
- tensorly-torch
- torch-harmonics
- opt-einsum
- h5py
- zarr
- scikit-optimize
- papermill