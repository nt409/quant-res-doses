import pandas as pd
import numpy as np

DEFAULT_I0 = float(
    pd.read_csv('../data/I0_value.csv')
    .loc[:, ['I0_value']]
    .iloc[0]
)

DEFAULT_BETA = float(
    pd.read_csv('../data/beta_value.csv')
    .loc[:, ['beta_median']]
    .iloc[0]
)

ALL_BETAS = np.asarray(
    pd.read_csv('../data/beta_sampled.csv').beta
)

MUTATION_SCALE = float(
    pd.read_csv('../data/mutation_scale.csv')
    .loc[:, ['mutation_scale']]
    .iloc[0]
)

MUTATION_PROP = (0.5 * (28 + 130) * 1e6) / (0.5 * (2.3 + 10.5) * 1e12)

DEFAULT_P = 0.1

DEFAULT_MUTATION_SCALE = DEFAULT_P * MUTATION_SCALE

FUNG_DECAY_RATE = 0.5 * (6.91e-3 + 1.11e-2)
