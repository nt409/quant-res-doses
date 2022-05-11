"""
Functions to run model - wrapped in run.py

NB need to document!
"""

import copy
import itertools

import numpy as np
from joblib import Memory
from tqdm import tqdm

from polymodel.params import PARAMS
from polymodel.simulator import Simulator
from polymodel.utils import (config_str_to_log, keys_from_config,
                             run_logger, yield_function)

memory = Memory('../joblib_cache/', verbose=1)

#
#
#


@memory.cache
def simulations_run(config,
                    I0_vec_run=None,
                    beta_vec_run=None,
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


@memory.cache
def variable_strategy_run(config, first, second):
    """Variable Strategy

    Parameters
    ----------
    config : Config
        see Config docs

    spray_first_n : int
        Application for first N years

    spray_second_n : int
        Application for second N years

    Returns
    -------
    model_output_list : list
        list of simulation outputs; one for each fungicide strategy

    Example
    -------
    >>>config_variable = Config(
    ...     type='variable',
    ...     n_k=100,
    ...     n_l=100,
    ...     host_on=[False],
    ... )
    >>>data = variable_strategy_run(config_variable, 2, 3)
    """
    return no_joblib_variable_strategy_run(config, first, second)


@memory.cache
def mutual_protection_run(config):
    """Mutual protection

    This function returns the disease severities if you take the host/fungicide
    to evolve as if both were on, but remove the protection offered by
    the other trait.

    Parameters
    ----------
    config : Config
        NB should only have sprays, host_on as follows:
        - sprays = [1] OR [2] OR [3]
        - host_on = [True, False]
        Should be of type 'multi' - see example below

    Returns
    -------
    yield_diff : dict
        keys like 
        - spray_Y2_host_Y_without_host
        each value an np.array with size n_its and n_years (need to check order)

    Example
    -------
    >>>config = Config(
    ...  n_k=100,
    ...  n_l=100,
    ...  disease_pressure='med',
    ...  host_growth=True,
    ...  cultivar='Hereford',
    ...  type='multi',
    ...  replace_cultivars=False,
    ...  sprays=[2],
    ...  host_on=[False],
    ...  n_iterations=5,
    ... )
    >>>data = mutual_protection_run(config)
    """
    return no_joblib_mutual_protection_run(config)


#
#
#

def no_joblib_simulations_run(
    config,
    I0_vec_run=None,
    beta_vec_run=None,
    verbose=True,
):
    """See docs above for joblib version"""

    if verbose:
        print('running simulation')

    run_logger.info(config_str_to_log(config))

    if beta_vec_run is None:
        beta_vec_run = [config.beta_single] * config.n_years

    if I0_vec_run is None:
        I0_vec_run = [config.I0_single] * config.n_years

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

        if verbose:
            run_logger.info('running single run:')
            run_logger.info(config_str_to_log(config_save))

        fungicide_on = False if (host and spray == 0) else True

        this_simulation = Simulator(
            host_plant_on=host,
            fungicide_on=fungicide_on,
            number_of_sprays=spray,
            config=config_save
        )

        key = keys_from_config(config_save)[0]

        model_output[key] = this_simulation.run_model(
            I0_vec=I0_vec_run,
            beta_vec=beta_vec_run,
        )

    return model_output


#
#
#

def no_joblib_variable_strategy_run(config, spray_first_n, spray_second_n):
    """See docs above for joblib version"""

    print('running variable')

    beta_vec_run = [config.beta_single] * config.n_years

    I0_vec_run = [config.I0_single] * config.n_years

    for host_on in config.host_on:

        run_logger.info('running variable strategy run:')
        run_logger.info(config_str_to_log(config))

        num_strats = config.n_years + 1
        model_output_list = [[]]*(num_strats)

        for ii in tqdm(range(num_strats)):
            spray_vec = (
                [spray_first_n]*ii +
                [spray_second_n]*(num_strats-1-ii)
            )

            spray_vec = np.asarray(spray_vec)

            this_simulation = Simulator(
                host_plant_on=host_on,
                fungicide_on=True,
                custom_sprays_vec=spray_vec,
                config=config
            )

            model_output_list[ii] = this_simulation.run_model(
                I0_vec=I0_vec_run,
                beta_vec=beta_vec_run,
            )

    return model_output_list


def no_joblib_multiple_run(config):
    """See docs for joblib version above"""

    print('running multiple runs')

    run_logger.info('running (multiple iterations)')
    run_logger.info(
        config_str_to_log(config) + f', its={config.n_iterations}'
    )

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


# --------------------------------------------------------------------


def no_joblib_mutual_protection_run(config):
    """See docs above for joblib version"""

    print('running mutual protection')

    multi_outputs = multiple_run(config)

    multi_output_keys = keys_from_config(config)

    yield_diff = {}

    for key in multi_output_keys:

        multi_output = multi_outputs[key]

        sev = {}

        sev['without_host'] = []
        sev['without_fung'] = []

        for trait, key_trait, fung_benefit in zip(
            ['host', 'fung'],
            ['host_', 'spray_'],
            [True, False]
        ):

            run_logger.info('running mutual protection:')
            run_logger.info(config_str_to_log(config))

            without_trait = f'without_{trait}'

            # need host/fung to have been on to have anything new to calculate
            trait_is_off = key.split(key_trait)[1].startswith('N')
            if trait_is_off:
                for i in range(len(multi_output)):
                    sev[without_trait].append(multi_output[i]['dis_sev'])

            else:
                sev[without_trait] = _find_protection(
                    config,
                    multi_output,
                    key,
                    fung_benefit=fung_benefit
                )

            # find yield
            yield_diff[key + '_' + without_trait] = (
                np.vectorize(yield_function)(sev[without_trait])
            )

    return yield_diff


def _find_protection(config, multi_output, key, fung_benefit):
    """Protective effect

    Effect on severities of switching host/fung off, but keeping fung/host dists

    Parameters
    ----------
    config : Config
        See Config docs.

    multi_output : list of dicts
        e.g. multi_outputs['key']

    key : _type_
        of form:
        - spray_N_host_N
        - spray_Y2_host_N

    fung_benefit : bool
        whether testing benefit to the fungicide or to the host

    Returns
    -------
    _type_
        _description_
    """

    print(key, fung_benefit)

    # testing benefit to the fungicide - switch host off
    if fung_benefit:
        simulation = Simulator(
            host_plant_on=False,
            fungicide_on=True,
            config=config
        )

        num_sprays = config.sprays[0]

        if key.startswith('spray_N'):
            num_sprays = 0

    # testing benefit to the host - 0 sprays
    else:
        if 'host_N' in key:
            simulation = Simulator(
                host_plant_on=False,
                fungicide_on=True,
                config=config
            )

        else:
            simulation = Simulator(
                host_plant_on=True,
                fungicide_on=False,
                config=config
            )

        num_sprays = 0

    sevs_out = []

    for i in range(len(multi_output)):
        soln = multi_output[i]

        infections_this_run = []

        betas = soln['beta_vec']
        I0s = soln['I0_vec']

        for year in range(len(betas)):

            _, _, _, _, total_infection = simulation.calculate_ode_soln(
                soln['fung_dists'][:, year],
                soln['host_dists'][:, year],
                I0s[year],
                betas[year],
                num_sprays
            )

            infections_this_run.append(total_infection[-1])

        sevs_out.append(infections_this_run)
    return sevs_out
