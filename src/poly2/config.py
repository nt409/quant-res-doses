"""
This file sets up the parameters for the polygenic model

Two setups:
- Config (one or two traits valid here)
- ConfigMixture (two fungicides)


And a single function:
print_str_repr
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


class Config:
    def __init__(
        self,
        #
        type=None,
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
        decay_rate=None,
        asymptote=None,
        #
        dose=None,
        #
        verbose=True,
    ):
        """Config for polygenic model

        There are various ways we want to run the model:
        - single run
        - multiple run (varying disease pressure)

        Then specify tactics for single/multi runs via sprays and host on

        Specify the scenario with disease pressure and cultivar

        Parameters
        ----------
        type : str, optional
            Can be:
            - 'single'
            - 'multi'
            by default None which gives 'single'

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

        mutation_scale_fung : float, optional
            Scaling for mutation (assume gaussian dispersal), by default 0

        mutation_scale_host : float, optional
            Scaling for mutation (assume gaussian dispersal), by default 0

        decay_rate : float, optional
            Fungicide decay rate if want =/= default, by default None

        dose : float, optional
            use dose * np.ones(n_years), by default None
            If None, use np.ones(n_years)

        verbose : bool, optional
            whether to print out summary of config, by default True

        Examples
        --------
        >>>single = Config(
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

        self.n_years = n_years

        #
        #
        # STRATEGY
        self.sprays = sprays

        self.fungicide_mixture = False

        self.n_k = n_k
        self.n_l = n_l

        self.decay_rate = decay_rate
        self.asymptote = asymptote

        if dose is None:
            self.doses = np.ones(self.n_years)
        else:
            self.doses = dose * np.ones(self.n_years)

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

        if type is None:
            self.type = 'single'
        else:
            self.type = type

        # self.I0_single = DEFAULT_I0
        self.I0s = DEFAULT_I0 * np.ones(self.n_years)

        if self.type == 'single':
            self.betas = DEFAULT_BETA * np.ones(self.n_years)

        elif self.type == 'multi':
            self.beta_multi = ALL_BETAS
            self.n_iterations = n_iterations

        if verbose:
            print_string_repr(self)
    #
    #

    def print_repr(self):
        print_string_repr(self)
        #
        #
        #
        #
        #
        #


class ConfigMixture:
    def __init__(
        self,
        #
        type=None,
        #
        sprays=None,
        #
        n_k=50,
        #
        n_iterations=None,
        n_years=15,
        #
        mutation_proportion=MUTATION_PROP,
        mutation_scale_fung=DEFAULT_MUTATION_SCALE,
        #
        decay_rate_A=None,
        decay_rate_B=None,
        #
        dose_A=None,
        dose_B=None,
        #
        verbose=True,
    ):
        """Config for polygenic model

        There are various ways we want to run the model:
        - single run
        - multiple run (varying disease pressure)

        Then specify tactics for single/multi runs via sprays and host on

        Specify the scenario with disease pressure and cultivar

        Parameters
        ----------
        type : str, optional
            Can be:
            - 'single'
            - 'multi'
            by default None which gives 'single'

        sprays : list of ints, optional
            will be passed into itertools.product with host_on
            e.g.
            [0,1,2,3] will check all spray possibilities
            by default None

        n_k : int, optional
            Number controlling fungicide distribution discretisation
            Suggest XX for final run and YY for quick run
            By default 50

        n_iterations : int, optional
            number of iterations if running multi, by default None

        n_years : int, optional
            number of years to run the model, by default 25

        mutation_proportion : float, optional
            Proportion of pathogen population that mutates.
            Between 0 and 1, by default 0

        mutation_scale_fung : float, optional
            Scaling for mutation (assume gaussian dispersal), by default 0

        decay_rate_A : float, optional
            Fungicide decay rate if want =/= default, by default None

        decay_rate_B : float, optional
            Fungicide decay rate if want =/= default, and using fungicide
            mixture, by default None

        dose_A : float, optional
            use dose * np.ones(n_years), by default None
            If None, use np.ones(n_years)

        dose_B : float, optional
            Fungicide B; 
            use dose_B * np.ones(n_years), by default None
            If None, use np.ones(n_years)

        verbose : bool, optional
            whether to print out summary of config, by default True

        Examples
        --------
        >>>single = Config(
        ... sprays=[2],
        ... dose_A=0.5,
        ... dose_B=0.25,
        ... )
        >>>multi = Config(
        ... type='multi',
        ... sprays=[2],
        ... dose_A=0.5,
        ... dose_B=0.25,
        ... )
        """

        self.n_years = n_years

        #
        #
        # STRATEGY
        self.sprays = sprays

        self.n_k = n_k

        self.decay_rate_A = decay_rate_A
        self.decay_rate_B = decay_rate_B

        if dose_A is None:
            self.doses_A = np.ones(self.n_years)
        else:
            self.doses_A = dose_A * np.ones(self.n_years)

        if dose_B is None:
            self.doses_B = np.ones(self.n_years)
        else:
            self.doses_B = dose_B * np.ones(self.n_years)

        self.host_on = [False]

        self.fungicide_mixture = True

        #
        #
        # PATHOGEN
        self.mutation_proportion = mutation_proportion

        self.mutation_scale_fung = mutation_scale_fung

        fit = pd.read_csv('../data/fitted.csv')

        assert len((
            fit.loc[(
                (np.isclose(fit.mutation_prop, mutation_proportion)) &
                (np.isclose(fit.mutation_scale_fung, mutation_scale_fung))
            )]
        )) == 2

        self.A_mu = float(fit.loc[lambda df: df.trait == 'Fungicide', 'mu'])
        self.A_b = float(fit.loc[lambda df: df.trait == 'Fungicide', 'b'])

        self.B_mu = float(fit.loc[lambda df: df.trait == 'Fungicide', 'mu'])
        self.B_b = float(fit.loc[lambda df: df.trait == 'Fungicide', 'b'])

        #
        #
        # SCENARIO

        if type is None:
            self.type = 'single'
        else:
            self.type = type

        self.I0s = DEFAULT_I0 * np.ones(self.n_years)

        if self.type == 'single':
            self.betas = DEFAULT_BETA * np.ones(self.n_years)

        elif self.type == 'multi':
            self.beta_multi = ALL_BETAS
            self.n_iterations = n_iterations

        if verbose:
            print_string_repr(self)
    #
    #

    def print_repr(self):
        print_string_repr(self)
        #
        #
        #
        #
        #
        #


def print_string_repr(obj):
    str_out = "CONFIG\n------\n"

    for key, item in sorted(vars(obj).items()):

        this_key = (
            f"{str(key)}={str(item)}"
            .replace('{', '')
            .replace('}', '')
            # .replace('[', '')
            # .replace(']', '')
            .replace("'", '')
            .replace(' ', ', ')
            .replace(':', '--')
            .replace('=', ' = ')
            .replace(',,', ',')
        )

        if len(this_key) >= 54:
            this_key = this_key[:50] + " ..."

        str_out += this_key + "\n"

    print(str_out)
