# Polygenic 2

This is code to accompany the paper: ['Coupling machine learning and epidemiological modelling to characterise optimal fungicide doses when fungicide resistance is partial or quantitative']().

The basic model is very similar to that found in [this repository](github.com/nt409/quantitative-resistance), as described in ['Modelling quantitative fungicide resistance and breakdown of resistant cultivars: designing integrated disease management strategies for Septoria of winter wheat'](biorxiv.org/content/10.1101/2022.08.10.503500v1.full).

It is a model of quantitative fungicide resistance, parameterised for septoria of winter wheat. The model is fitted to field data.

### Data

The default model parameter values are found in `data`.

## Code

### Scans

The model is implemented in python.

In the paper ['Coupling machine learning and epidemiological modelling to characterise optimal fungicide doses when fungicide resistance is partial or quantitative'](), we describe the model found in `src/poly2`.

The most important file in this folder is `simulator.py`, which contains the classes `SimulatorOneTrait` and `SimulatorAsymptote` which are the partial resistance Type 2 and partial resistance Type 1 models respectively. These have docstrings describing their use and configuration.

### Scans

We ran two larger ensemble runs where the mutation and fungicide paramters between runs. The code that generated the outputs of the scans is found in `src/cluster`. We ran two scans - one each for partial resistance Type 1 (`src/cluster/scan_all.submit`) and Type 2 (`src/cluster/scan_asymp.submit`).

### Gradient-boosted trees models

The XGBoost models as described in the paper are fitted and optimised in:

- `cluster/hyper_all.submit`
- `cluster/hyper_asymp.submit`
- `cluster/hyper_cumulative.submit`
- `cluster/hyper_Y10.submit`

All of these are run on the HPC. Some analysis and postprocessing was done using the notebooks in the `notebooks` folder.

### Shapley values

Shapley values are found using `src/poly2/shap.py`.

### Plotting

Most of the plotting for the actual paper was done in the notebooks in the `notebooks` folder. A few helper functions can be found in `src/plots2`, although you may prefer to write your own custom plotting functions.
