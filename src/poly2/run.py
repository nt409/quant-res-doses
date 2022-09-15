"""
Convenient wrappers for running model, can instead directly run
SimulatorOneTrait or SimulatorBothTraits

functions:
- single_run (caches result)
- no_joblib_single_run (doesn't cache result)
- multiple_run (doesn't cache result)
- no_joblib_multiple_run (doesn't cache result)
- get_data_from_pars
"""

import copy
import itertools

import numpy as np
from joblib import Memory
from tqdm import tqdm

import pandas as pd

from poly2.config import Config
from poly2.consts import FUNG_DECAY_RATE, MUTATION_PROP, MUTATION_SCALE
from poly2.params import PARAMS
from poly2.simulator import SimulatorBothTraits, SimulatorMixture, SimulatorOneTrait
from poly2.utils import keys_from_config


memory = Memory('../joblib_cache/', verbose=1)

#
#
#


@memory.cache
def single_run(config, verbose=True):
    """Run polygenic model

    Parameters
    ----------
    config : Config (see config classes)
        _description_

    verbose : bool, optional
        Whether to print 'running simulation' or not - fine in general but
        annoying in the 'multi' case, by default True

    Returns
    -------
    model_output : dict
        key is of the form:
        - spray_N_host_N
        - spray_1_host_Y

    Example
    -------
    >>>config_single = Config(
    ...     type='single',
    ...     sprays=[2],
    ...     host_on=[False],
    ... )
    >>>data = simulations_run(config_single)
    """
    return no_joblib_single_run(config, verbose)


@memory.cache
def multiple_run(config):
    """Multiple runs

    Uses varying beta.

    Parameters
    ----------
    config : Config
        see Config docs

    Returns
    -------
    dict_of_lists : dict of lists
        Each list contains model outputs

    Example
    -------
    >>>config = Config(
    ...     type='multi',
    ...     n_k=100,
    ...     n_l=100,
    ...     sprays=[2],
    ...     host_on=[False],
    ...     n_iterations=10,
    ... )
    >>>data = multiple_run(config)
    """
    return no_joblib_multiple_run(config)

#
#
#


def no_joblib_single_run(config, verbose=True):
    """See docs above for joblib version"""

    if verbose:
        print('running simulation')

    conf_use = copy.deepcopy(config)

    model_output = {key: [] for key in PARAMS.strategy_mapping.keys()}

    for spray, host in itertools.product(
        config.sprays,
        config.host_on
    ):

        conf_use.sprays = [spray]
        conf_use.host_on = [host]

        if conf_use.fungicide_mixture:

            this_simulation = SimulatorMixture(
                config=conf_use,
                number_of_sprays=spray,
            )

        else:

            fungicide_on = False if (host and spray == 0) else True

            if host and fungicide_on:
                this_simulation = SimulatorBothTraits(
                    config=conf_use,
                    number_of_sprays=spray,
                )

            else:
                this_simulation = SimulatorOneTrait(
                    config=conf_use,
                    fungicide_on=fungicide_on,
                    host_plant_on=host,
                    number_of_sprays=spray,
                )

        key = keys_from_config(conf_use)[0]

        model_output[key] = this_simulation.run_model()

    return model_output


def no_joblib_multiple_run(config):
    """See docs for joblib version above"""

    print('running multiple runs')

    #
    #

    n_total = config.n_iterations * config.n_years

    betas_sample_from = np.reshape(
        config.beta_multi[:n_total], (config.n_iterations, config.n_years)
    )

    conf_use = copy.deepcopy(config)
    #
    #

    dict_of_lists = {key: [] for key in PARAMS.strategy_mapping.keys()}

    #
    #

    for ii in tqdm(range(config.n_iterations)):

        conf_use.betas = np.asarray(betas_sample_from[ii, :])

        model_output_dict = no_joblib_single_run(
            conf_use,
            verbose=False
        )

        for key in PARAMS.strategy_mapping.keys():
            dict_of_lists[key].append(model_output_dict[key])

    return dict_of_lists


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
