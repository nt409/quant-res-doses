"""
This file sets up the parameters for the polygenic model
"""

import numpy as np
import pandas as pd

from poly2.consts import (
    ALL_BETAS,
    DEFAULT_BETA,
    DEFAULT_I0,
    DEFAULT_MUTATION_SCALE,
    MUTATION_PROP
)


# * TOC

# Config
# Fitting

# ------------------------------------------------------------


class Config:
    def __init__(
        self,
        #
        type: str,
        #
        sprays=None,
        host_on=None,
        #
        n_k=50,
        n_l=50,
        #
        n_iterations=None,
        n_years=15,
        #
        replace_cultivars=False,
        #
        mutation_proportion=MUTATION_PROP,
        mutation_scale_host=DEFAULT_MUTATION_SCALE,
        mutation_scale_fung=DEFAULT_MUTATION_SCALE,
        #
        verbose=True,
    ):
        """Config for polygenic model

        There are various ways we want to run the model:
        - single run
        - multiple run (varying disease pressure)
        - variable run (varying fungicide strategy)
        - mutual protection run

        Then specify tactics for single/multi runs via sprays and host on

        Specify the scenario with disease pressure and cultivar

        Parameters
        ----------
        type : str
            Can be:
            - 'single'
            - 'multi'
            - 'variable'

        sprays : list of ints, optional
            will be passed into itertools.product with host_on
            e.g.
            [0,1,2,3] will check all spray possibilities
            by default None

        host_on : list of booleans, optional
            will be passed into itertools.product with sprays
            e.g.
            [False] will run without host protection
            [True] will run with host protection
            [False, True] will run both
            by default None

        n_k : int, optional
            Number controlling fungicide distribution discretisation
            Suggest XX for final run and YY for quick run
            By default 50

        n_l : int, optional
            Number controlling fungicide distribution discretisation
            Suggest XX for final run and YY for quick run
            By default 50

        n_iterations : int, optional
            number of iterations if running multi, by default None

        n_years : int, optional
            number of years to run the model, by default 25

        replace_cultivars : bool, array, optional
            whether or not to replace cultivars, by default False.
            If want to, specify an array of booleans for whether to replace at
            of year 0, 1, ... N-1

        mutation_proportion : float, optional
            Proportion of pathogen population that mutates.
            Between 0 and 1, by default 0

        mutation_scale_fung : float/bool, optional
            Scaling for mutation (assume gaussian dispersal), by default 0

        mutation_scale_host : float/bool, optional
            Scaling for mutation (assume gaussian dispersal), by default 0

        verbose : bool, optional
            whether to print out summary of config, by default True

        Examples
        --------
        >>>single = Config(
        ... type='single',
        ... sprays=[2],
        ... host_on=[False],
        ... n_k=100,
        ... n_l=100
        ... )
        >>>multi = Config(
        ... type='multi',
        ... sprays=[0,2],
        ... host_on=[False],
        ... n_iterations=10
        ... )
        """

        self.n_k = n_k
        self.n_l = n_l

        self.n_iterations = n_iterations

        self.n_years = n_years

        #
        #
        # STRATEGY
        self.sprays = sprays

        self.host_on = host_on

        if replace_cultivars is not False:
            self.replace_cultivars = replace_cultivars
        else:
            self.replace_cultivars = None

        #
        #
        # PATHOGEN
        self.mutation_proportion = mutation_proportion

        self.mutation_scale_host = mutation_scale_host
        self.mutation_scale_fung = mutation_scale_fung

        fit = pd.read_csv('../data/fitted.csv')

        assert len((
            fit.loc[(
                (np.isclose(fit.mutation_prop, mutation_proportion)) &
                (np.isclose(fit.mutation_scale_fung, mutation_scale_fung)) &
                (np.isclose(fit.mutation_scale_host, mutation_scale_host))
            )]
        )) == 2

        self.k_mu = float(fit.loc[lambda df: df.trait == 'Fungicide', 'mu'])
        self.k_b = float(fit.loc[lambda df: df.trait == 'Fungicide', 'b'])

        self.l_mu = float(fit.loc[lambda df: df.trait == 'Mariboss', 'mu'])
        self.l_b = float(fit.loc[lambda df: df.trait == 'Mariboss', 'b'])

        #
        #
        # SCENARIO

        self.type = type

        self.I0_single = DEFAULT_I0

        if type in ['single', 'variable']:
            self.beta_single = DEFAULT_BETA

        elif type == 'multi':
            self.beta_multi = ALL_BETAS

        if verbose:
            self.print_string_repr()

        #
        #
        #
        #
        #
        #

    def print_string_repr(self):
        str_out = "CONFIG\n------\n"

        for key, item in sorted(vars(self).items()):
            if len(str(item)) > 50:
                item_str = f"{str(item)[:50]}..."
            else:
                item_str = f"{str(item)}"

            str_out += f"{str(key)}={item_str}\n"

        str_out = (
            str_out
            .replace('{', '')
            .replace('}', '')
            .replace('[', '')
            .replace(']', '')
            .replace("'", '')
            .replace(' ', ', ')
            .replace(':', '--')
            .replace('=', ' = ')
            .replace(',,', ',')
        )

        print(str_out)

        # return str_out
