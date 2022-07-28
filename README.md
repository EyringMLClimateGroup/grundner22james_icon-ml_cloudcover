# iconml_clc
A neural network based cloud cover parameterization for the ICON-A model. <br>
This repository is part of our paper submitted to JAMES. 

It contains the neural network and analysis code, and an extract from the training data.

Corresponding DOI: [![DOI](https://zenodo.org/badge/436660284.svg)](https://zenodo.org/badge/latestdoi/436660284)

---------------
**Structure of the six folders corresponding to the neural networks**

There are essentially three notebooks in each of the folders corresponding to each of the models <br>
`n1*, n2*, n3*, q1*, q2*, q3*`:

1. In **preprocessing** notebooks: <br>
We read the coarse-grained data from nc-files, preprocess it, and store it in npy-files for faster access. The NARVAL data is standardized w.r.t. mean and variance of the training dataset before it was stored. The QUBICC data is not standardized and split up yet, due to the three-fold cross-validation split. A fraction of the coarse-grained and/or preprocessed data is located in /extract_from_the_data.

2. In **commence training** notebooks: <br>
We load the data from the npy-files, standardize and split it up for the QUBICC models, and train our neural networks on it.

3. The **evaluation** notebooks <br>
Contain the evaluation of the models. But note that the generated figures were not used in the paper directly (see overview_figures.txt).
Much of the QUBICC model evaluation is done in /additional_content/plots_offline_paper/qubicc_model_plots.ipynb.

---------------

---------------
**Old model names**

The neighborhood-based model from the paper is often called region-based model in the code. <br>
More specifically, this is the *new name <- old name* mapping for every model:

- n1_cell_based_narval_r2b4 <- grid_cell_based_v3
- n2_column_based_narval_r2b4 <- grid_column_based
- n3_neighborhood_based_narval_r2b4 <- region_based
- q1_cell_based_qubicc_r2b5 <- grid_cell_based_QUBICC_R02B05
- q2_column_based_qubicc_r2b5 <- grid_column_based_QUBICC_R02B05
- q3_neighborhood_based_qubicc_r2b5 <- region_based_one_nn_R02B05

