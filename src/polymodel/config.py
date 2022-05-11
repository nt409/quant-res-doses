"""
This file sets up the parameters for the polygenic model
"""

import numpy as np
import pandas as pd

from polymodel.consts import ALL_BETAS, DEFAULT_BETA, DEFAULT_I0


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
        cultivar=None,
        #
        n_iterations=None,
        n_years=25,
        #
        replace_cultivars=False,
        #
        mutation_proportion=0,
        mutation_scale_host=None,
        mutation_scale_fung=None,
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

        # disease_pressure : str, optional
        #     Takes values in:
        #     - 'low'
        #     - 'med'
        #     - 'high'
        #     This gets mapped to a location.
        #     By default None, which we set to 'med'.

        cultivar : str, optional
            name of cultivar, e.g.:
            - 'Hereford'
            - 'Mariboss'
            , by default None (which we set to 'Mariboss')

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
            Scaling for mutation (assume exponetial dispersal), by default 0

        mutation_scale_host : float/bool, optional
            Scaling for mutation (assume exponetial dispersal), by default 0

        verbose : bool, optional
            whether to print out summary of config, by default True

        Examples
        --------
        >>>single = Config(type='single', sprays=[2], host_on=[False], n_k=100,
        ... n_l=100)
        >>>multi = Config(type='multi', sprays=[0,2], host_on=[False],
        ... n_iterations=10)
        >>>variable = Config(type='variable', host_on=[False])
        """

        self.n_k = n_k
        self.n_l = n_l

        self.n_iterations = n_iterations

        self.n_years = n_years

        if cultivar is None:
            self.cultivar = 'Mariboss'
        else:
            self.cultivar = cultivar

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
        # if mutation_proportion == 0:
        #     self.mutation_on = False
        # else:
        #     self.mutation_on = True

        self.mutation_proportion = mutation_proportion

        self.mutation_scale_host = mutation_scale_host
        self.mutation_scale_fung = mutation_scale_fung

        fitted_df = pd.read_csv('../data/03_model_inputs/fitted.csv')

        filt_by_mut = (
            fitted_df
            .loc[
                np.isclose(fitted_df.mutation_prop,
                           mutation_proportion)
            ]
        )

        #
        #

        # fung dist params:
        fung_frame = (
            filt_by_mut
            .loc[lambda df: (
                (df['trait'] == 'Fungicide') &
                (
                    np.isclose(df.mutation_scale_fung,
                               mutation_scale_fung)
                )
            )
            ]
        )

        if len(fung_frame) == 1:
            self.k_mu = float(fung_frame.mu)
            self.k_b = float(fung_frame.b)

        elif len(fung_frame) > 1:
            print(f'WARNING: {len(fung_frame)=}')
            self.k_mu = float(fung_frame.mu.iloc[0])
            self.k_b = float(fung_frame.b.iloc[0])

        else:
            print(f'WARNING: {len(fung_frame)=}')
            self.k_mu = None
            self.k_b = None

        #
        #

        # host dist params:
        if self.cultivar in list(filt_by_mut.trait):
            host_frame = filt_by_mut.loc[lambda df: (
                (df.trait == self.cultivar) &
                (
                    np.isclose(df.mutation_scale_host,
                               mutation_scale_host)
                )
            )]

            if len(host_frame) > 1:
                print(f'WARNING: {len(host_frame)=}')

                self.l_mu = float(host_frame.mu.iloc[0])
                self.l_b = float(host_frame.b.iloc[0])

            elif len(host_frame) == 1:

                self.l_mu = float(host_frame.mu)
                self.l_b = float(host_frame.b)

            else:
                print(f'WARNING: {len(host_frame)=}')
                self.l_mu = None
                self.l_b = None
        else:
            print(f'WARNING: {self.cultivar=}, {list(filt_by_mut.trait)=}')

            self.l_mu = None
            self.l_b = None

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
        # for repr, conf_str_gen see below
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
