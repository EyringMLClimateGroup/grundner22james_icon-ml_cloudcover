# ML-based cloud cover parameterization for the ICON model
A neural network based cloud cover parameterization for the ICON-A model. <br>
This repository contains the neural network and analysis code, and an extract from the training data.

It is part of our paper published in JAMES:
> Grundner, A., Beucler, T., Gentine, P., Iglesias-Suarez, F., Giorgetta, M. A., & Eyring, V. (2022). Deep learning based cloud cover parameterization for ICON. Journal of Advances in Modeling Earth Systems, 14, e2021MS002959. https://doi.org/10.1029/2021MS002959 

Link to the pre-print: [arXiv:2112.11317](https://arxiv.org/abs/2112.11317)

Corresponding DOI: [![DOI](https://zenodo.org/badge/436660284.svg)](https://zenodo.org/badge/latestdoi/436660284)

---------------
**Usage**

To install or re-run parts of the code, please install the environment by executing *conda env create -f clouds113_env_with_shap.yml*. 
The Sherpa package is not included in the environment file and would, if required, need to be installed separately (*pip install sherpa*, version 4.12.1). We provide the data for [quickstart.ipynb](../master/quickstart.ipynb), which can directly be executed. Other than that a fraction of the data used to train the networks is located in [extract_from_the_data](../master/extract_from_the_data) and described in [extract_from_the_data/README](../master/extract_from_the_data/README).

---------------
**Structure of the six folders corresponding to the neural networks**

There are essentially three notebooks in each of the folders corresponding to each of the models <br>
`n1*, n2*, n3*, q1*, q2*, q3*`:

1. In **preprocessing** notebooks: <br>
We read the coarse-grained data from nc-files, preprocess it, and store it in npy-files for faster access. The NARVAL data is standardized w.r.t. mean and variance of the training dataset before it was stored. The QUBICC data is not standardized and split up yet, due to the three-fold cross-validation split. A fraction of the coarse-grained and/or preprocessed data is located in /extract_from_the_data.

2. In **commence training** notebooks: <br>
We load the data from the npy-files, standardize and split it up for the QUBICC models, and train our neural networks on it.

3. The **evaluation** notebooks <br>
Contain the evaluation of the models. But note that the generated figures were not used in the paper directly (see [overview_figures](../master/overview_figures)).
Much of the QUBICC model evaluation is done in [additional_content/plots_offline_paper/qubicc_models_plots.ipynb](../master/additional_content/plots_offline_paper/qubicc_models_plots.ipynb).

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

