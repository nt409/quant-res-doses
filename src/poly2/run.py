"""
Functions to run model - wrapped in run.py

Convenient wrapper for running model, can instead directly run
SimulatorOneTrait or SimulatorBothTraits
"""

import copy
import itertools

import numpy as np
from joblib import Memory
from tqdm import tqdm

from poly2.params import PARAMS
from poly2.simulator import SimulatorBothTraits, SimulatorOneTrait
from poly2.utils import keys_from_config


memory = Memory('../joblib_cache/', verbose=1)

#
#
#


@memory.cache
def simulations_run(config,
                    I0_vec_run=None,
                    beta_vec_run=None,
                    doses=None,
                    verbose=True):
    """Run polygenic model

    Parameters
    ----------
    config : Config (see config classes)
        _description_

    I0_vec_run : list/np.array, optional
        I0 values, overrides config.I0_single/config.I0_multi, by default None

    beta_vec_run : list/np.array, optional
        Beta values, overrides config.beta_single, by default None

    doses : list/np.array, optional
        Doses, one per year, by default None

    verbose : bool, optional
        Whether to print 'running simulation' or not - fine in general but
        annoying in the 'multi' case, by default True

    Returns
    -------
    model_output : dict
        key is of the form:
        - spray_N_host_N
        - spray_Y1_host_Y

    Example
    -------
    >>>config_single = Config(
    ...     type='single',
    ...     sprays=[2],
    ...     host_on=[False],
    ... )
    >>>data = simulations_run(config_single)
    """
    return no_joblib_simulations_run(config,
                                     I0_vec_run,
                                     beta_vec_run,
                                     doses,
                                     verbose)


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


def no_joblib_simulations_run(
    config,
    I0_vec_run=None,
    beta_vec_run=None,
    doses_run=None,
    verbose=True,
):
    """See docs above for joblib version"""

    if verbose:
        print('running simulation')

    if beta_vec_run is None:
        beta_vec_run = config.beta_single * np.ones(config.n_years)

    if I0_vec_run is None:
        I0_vec_run = config.I0_single * np.ones(config.n_years)

    if doses_run is None:
        doses_run = np.ones(len(beta_vec_run))

    config_save = copy.deepcopy(config)

    model_output = {}
    for key in PARAMS.strategy_mapping.keys():
        model_output[key] = []

    for spray, host in itertools.product(
        config.sprays,
        config.host_on
    ):

        config_save.sprays = [spray]
        config_save.host_on = [host]

        fungicide_on = False if (host and spray == 0) else True

        if host and fungicide_on:
            this_simulation = SimulatorBothTraits(
                config=config_save,
                number_of_sprays=spray,
            )

        else:
            this_simulation = SimulatorOneTrait(
                config=config_save,
                fungicide_on=fungicide_on,
                host_plant_on=host,
                number_of_sprays=spray,
            )

        key = keys_from_config(config_save)[0]

        model_output[key] = this_simulation.run_model(
            I0_vec=I0_vec_run,
            beta_vec=beta_vec_run,
            doses=doses_run,
        )

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

    #
    #

    dict_of_lists = {}
    for key in PARAMS.strategy_mapping.keys():
        dict_of_lists[key] = []

    #
    #

    for ii in tqdm(range(config.n_iterations)):

        beta_vec_run = np.asarray(betas_sample_from[ii, :])
        I0_vec_run = config.I0_single * np.ones(len(beta_vec_run))

        model_output_dict = no_joblib_simulations_run(
            config,
            I0_vec_run,
            beta_vec_run,
            verbose=False
        )

        for key in PARAMS.strategy_mapping.keys():
            dict_of_lists[key].append(model_output_dict[key])

    return dict_of_lists
