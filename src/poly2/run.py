"""
Convenient functions to get specific data from params

functions:

- get_data_from_pars
- get_dists_from_pars
- get_config_from_pars
"""

import numpy as np
from tqdm import tqdm

import pandas as pd

from poly2.config import Config
from poly2.consts import FUNG_DECAY_RATE, MUTATION_PROP, MUTATION_SCALE
from poly2.simulator import SimulatorOneTrait


#
#
#


def get_data_from_pars(pars, n_d=10):
    """Get line df from parameter dictionary

    Parameters
    ----------
    pars : dict
        keys:
        - mu : float
        - b : float
        - asymp : float
        - dec_rate : float
        - m_prop : float
        - m_scale : float

    n_d : int, optional
        number of doses, by default 10

    Returns
    -------
    line_df : pd.DataFrame
        keys:
        - dose
        - yld
        - fung_mean
        - year
    """

    cf = get_config_from_pars(pars)

    line_df = pd.DataFrame()

    for dose in tqdm(np.linspace(0.1, 1, n_d)):

        cf.doses = dose*np.ones(cf.n_years)

        sim = SimulatorOneTrait(cf)

        data = sim.run_model()

        doses = np.repeat(dose, cf.n_years+1)

        ylds = np.concatenate([
            [np.nan],
            data['yield_vec']
        ])

        tmp = pd.DataFrame(
            dict(
                dose=doses,
                yld=ylds,
                fung_mean=data['fung_mean'],
                year=np.arange(0, 1+cf.n_years),
            )
        )

        line_df = pd.concat([line_df, tmp])

    return line_df


def get_dists_from_pars(pars, n_d=10):
    """Get line df from parameter dictionary

    Parameters
    ----------
    pars : dict
        keys:
        - mu : float
        - b : float
        - asymp : float
        - dec_rate : float
        - m_prop : float
        - m_scale : float

    n_d : int, optional
        number of doses, by default 10

    Returns
    -------
    df_out : pd.DataFrame
        keys:
        - k
        - density
        - year
        - dose
    """

    cf = get_config_from_pars(pars)

    df_out = pd.DataFrame()

    for dose in tqdm(np.linspace(0.1, 1, n_d)):

        cf.doses = dose*np.ones(cf.n_years)

        sim = SimulatorOneTrait(cf)

        data = sim.run_model()

        for year in [1, 10, 20, 30]:
            tmp = pd.DataFrame(
                dict(
                    k=data['k_vec'],
                    density=data['fung_dists'][:, year-1],
                )
            ).assign(year=year, dose=dose)

            df_out = pd.concat([df_out, tmp])

    return df_out


def get_config_from_pars(pars):

    cf = Config(
        n_years=35,
        n_k=300,
        verbose=False
    )

    if isinstance(pars, dict):
        cf.k_mu = pars['mu']
        cf.k_b = pars['b']
        cf.asymptote = pars['asymp']
        cf.decay_rate = FUNG_DECAY_RATE * pars['dec_rate']
        cf.mutation_proportion = MUTATION_PROP * pars['m_prop']
        cf.mutation_scale_fung = MUTATION_SCALE * pars['m_scale']
    else:
        cf.k_mu = pars.mu.values[0]
        cf.k_b = pars.b.values[0]
        cf.asymptote = pars.asymp.values[0]
        cf.decay_rate = FUNG_DECAY_RATE * pars.dec_rate.values[0]
        cf.mutation_proportion = MUTATION_PROP * pars.m_prop.values[0]
        cf.mutation_scale_fung = MUTATION_SCALE * pars.m_scale.values[0]

    return cf
