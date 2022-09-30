# Source code

## `poly2`

`poly2` contains:

- `simulator` - has the model(s)
- `utils` - many utility functions
- `config` - has the model configuration class(es)
- `consts/params` - has some important model parameters
- `shap` - for getting shapley values
- `run` - some functions for running the model

## `cluster`

`cluster` contains:

- `fig2*` - files like this used to generate the data for figure 2
- `hyper*` - files like this used to find optimal hyperparameters for the
  different XGBoost models
- `scan_all*` - files like this used to generate the main parameter scan
- `scan_asymp*` - files like this used to generate the partial resistance type
  1 parameter scan (resistance characterised by the asymptote)

## `plots2`

`plots2` contains:

- `fns` - some helper functions for plotting
- `consts` - some constants used when plotting
