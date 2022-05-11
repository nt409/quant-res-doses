import pandas as pd
import numpy as np

DEFAULT_I0 = float(
    pd.read_csv('../data/03_model_inputs/I0_value.csv')
    .loc[:, ['I0_value']]
    .iloc[0]
)


LAMBDA_FITTED = float(
    pd.read_csv('../data/03_model_inputs/lambda_fitted.csv')
    .loc[:, ['lambda_fitted']]
    .iloc[0]
)

DEFAULT_BETA = float(
    pd.read_csv('../data/03_model_inputs/beta_value.csv')
    .loc[:, ['beta_median']]
    .iloc[0]
)

ALL_BETAS = np.asarray(
    pd.read_csv('../data/03_model_inputs/beta_sampled.csv').beta
)

FUNG_MUTATION_SCALE = float(
    pd.read_csv('../data/03_model_inputs/fung_mutation_scale.csv')
    .loc[:, ['fung_mutation_scale']]
    .iloc[0]
)

HOST_MUTATION_SCALE = float(
    pd.read_csv('../data/03_model_inputs/host_mutation_scale.csv')
    .loc[:, ['host_mutation_scale']]
    .iloc[0]
)

# From Alexey paper:
# - 28 to 130 million pathogen spores carrying adaptive mutations to counteract fungicides and resistant cultivars
# - 2.3 to 10.5 trillion pycnidiospores per hectare
# Used to use the highest estimate:
# MUTATION_PROP = (130 * 1e6) / (2.3 * 1e12)
# Now using this estimate:
MUTATION_PROP = (0.5 * (28 + 130) * 1e6) / (0.5 * (2.3 + 10.5) * 1e12)

DEFAULT_P = 0.1

# FUNG_DECAY_RATE = 6.91e-3
FUNG_DECAY_RATE = 0.5 * (6.91e-3 + 1.11e-2)
# FUNG_DECAY_RATE = 1e-2
# FUNG_DECAY_RATE = 1.11e-2


# TRAIN_TEST_SPLIT_PROPORTION = 0.75
TRAIN_TEST_SPLIT_PROPORTION = 2/3
# TRAIN_TEST_SPLIT_PROPORTION = 0.75
# TRAIN_TEST_SPLIT_PROPORTION = 0.8
# TRAIN_TEST_SPLIT_PROPORTION = 1
