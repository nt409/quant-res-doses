import copy
from datetime import datetime

import numpy as np
import pandas as pd

from polymodel.simulator import Simulator
from polymodel.utils import find_beta_vectorised, initial_point_distribution
from polymodel.consts import TRAIN_TEST_SPLIT_PROPORTION


def score_for_this_df_weighted(df, control):
    """Get score for control model output relative to control data

    Weighted by N points per year

    Parameters
    ----------
    df : pd.DataFrame
        Columns:
        - year
        - data_control
        - n_data

    control : np.array
        array of model output (control) from first year to Nth

    Returns
    -------
    score : float
        model score - sum of squared residuals, weighted by N years, to 
        minimise
    """

    model_df = (
        pd.DataFrame(dict(model_control=control))
        .assign(year=np.arange(df.year.min(), df.year.max()+1))
    )

    results = (

        df.set_index('year')

        .join(model_df.set_index('year'))

        .assign(
            residuals=lambda x: (
                (x.model_control - x.data_control)**2
            ),

            weighted=lambda x: x.residuals * x.n_data,

        )
    )

    score = results.weighted.sum()

    return score


def score_for_this_df(df, control):
    """Get score for control model output relative to control data

    Parameters
    ----------
    df : pd.DataFrame
        Columns:
        - year
        - data_control

    control : np.array
        array of model output (control) from first year to Nth

    Returns
    -------
    score : float
        model score - sum of squared residuals - to minimise
    """

    model_df = (
        pd.DataFrame(dict(model_control=control))
        .assign(year=np.arange(df.year.min(), df.year.max()+1))
    )

    results = (

        df.set_index('year')

        .join(model_df.set_index('year'))

        .assign(
            residuals=lambda x: (
                (x.model_control - x.data_control)**2
            ),

        )
    )

    score = results.residuals.sum()

    return score


class HostObjective:
    def __init__(self, config) -> None:
        """Fit host distribution parameters

        Parameters
        ----------

        config : Config
            See Config docs - type 'single'

        """

        self.df = (
            pd.read_csv(
                '../data/03_model_inputs/control_host.csv'
            )
            .iloc[:, 1:]
            .loc[lambda df: df.cultivar == config.cultivar]
            .loc[lambda df: (
                df.year <= (
                    df.year.min() - 1
                    + TRAIN_TEST_SPLIT_PROPORTION *
                    (1 + df.year.max() - df.year.min())
                )
            )
            ]
            .rename(columns={
                'control': 'data_control',
                'min_num': 'n_data',
            })
        )

        input_severity_df = (
            pd.read_csv(
                '../data/03_model_inputs/input_severities_host.csv'
            )
            .iloc[:, 1:]
            .loc[lambda df: df.cultivar == config.cultivar]
            .loc[lambda df: df.year <= self.df.year.max()]
        )

        self.input_severities = 0.01*np.array(input_severity_df.worst_stb)

        n_years = input_severity_df.shape[0]

        self.I0s = [config.I0_single] * n_years

        self.betas = find_beta_vectorised(
            self.input_severities,
            config.I0_single
        )

        self.config = config

    #
    #

    def __call__(self, trial):

        if trial is None:
            params = {'mu': 0.5, 'b': 8}
        else:
            params = self.get_params(trial)

        control = self.run_model(params)

        return score_for_this_df_weighted(self.df, control)

    #
    #

    def run_model(self, params):

        NOT_USED_NUM = 0.5

        sevs_trait_off = self.input_severities

        output_trait_on = (
            Simulator(
                host_plant_on=True,
                fungicide_on=False,
                k_mu=NOT_USED_NUM,
                k_b=NOT_USED_NUM,
                l_mu=params['mu'],
                l_b=params['b'],
                config=self.config
            )
            .run_model(
                I0_vec=self.I0s,
                beta_vec=self.betas
            )
        )

        sevs_trait_on = np.asarray(output_trait_on['dis_sev'])

        control = 100 * (1 - sevs_trait_on / sevs_trait_off)

        return control

    #
    #

    def get_params(self, trial):
        params = {
            "mu": trial.suggest_float("mu", 1e-6, 1),
            "b": trial.suggest_float("b", 1e-6, 20),
        }
        return params

#
#
#
#
#


class FungicideObjective:
    def __init__(self, config) -> None:
        """Fit fungicide distribution parameters

        Parameters
        ----------
        config : Config
            See Config docs - type 'single'

        """

        self.df = (
            pd.read_csv(
                '../data/03_model_inputs/control_prothio_with_uncertainty.csv'
            )
            .loc[lambda df: (
                df.year <= (
                    df.year.min() - 1
                    + TRAIN_TEST_SPLIT_PROPORTION *
                    (1 + df.year.max() - df.year.min())
                )
            )
            ]
            .rename(columns={
                'control_with_noise_t': 'data_control',
            })
        )

        n_years = len(self.df.year.unique())

        input_severities = [0.37] * n_years

        self.input_severities = input_severities

        self.I0s = [config.I0_single] * n_years

        self.betas = find_beta_vectorised(input_severities, config.I0_single)

        self.config = config

    #
    #

    def __call__(self, trial):
        if trial is None:
            params = {'mu': 0.5, 'shape': 8}
        else:
            params = self.get_params(trial)

        control = self.run_model(params)

        return score_for_this_df(self.df, control)

    #
    #

    def run_model(self, params):

        NOT_USED_NUM = 0.5

        sevs_trait_off = np.asarray(self.input_severities)

        output_trait_on = (
            Simulator(
                host_plant_on=False,
                fungicide_on=True,
                number_of_sprays=1,
                l_mu=NOT_USED_NUM,
                l_b=NOT_USED_NUM,
                k_mu=params['mu'],
                k_b=params['b'],
                config=self.config
            )
            .run_model(
                I0_vec=self.I0s,
                beta_vec=self.betas
            )
        )

        sevs_trait_on = np.asarray(output_trait_on['dis_sev'])

        control = 100 * (1 - sevs_trait_on / sevs_trait_off)

        return control

    #
    #

    def get_params(self, trial):
        params = {
            # GAMMA
            "mu": trial.suggest_float("mu", 1e-2, 40),
            "b": trial.suggest_float("b", 1e-3, 20, log=True),

            # BETA
            # "mu": trial.suggest_float("mu", 1e-6, 1, log=True),
            # "b": trial.suggest_float("b", 1e-4, 30, log=True),
        }
        return params


#
#
#
#
#

class FungicideObjectiveAppendix:
    def __init__(self, config) -> None:
        """Fit fungicide distribution parameters

        As per FungicideObjective but don't split into train/test, use all data

        Parameters
        ----------
        config : Config
            See Config docs - type 'single'

        """

        self.df = (
            pd.read_csv(
                '../data/03_model_inputs/control_prothio_with_uncertainty.csv'
            )
            .rename(columns={
                'control_with_noise_t': 'data_control',
            })
        )

        n_years = len(self.df.year.unique())

        input_severities = [0.37] * n_years

        self.input_severities = input_severities

        self.I0s = [config.I0_single] * n_years

        self.betas = find_beta_vectorised(input_severities, config.I0_single)

        self.config = config

    #
    #

    def __call__(self, trial):
        if trial is None:
            params = {'mu': 0.5, 'shape': 8}
        else:
            params = self.get_params(trial)

        control = self.run_model(params)

        return score_for_this_df(self.df, control)

    #
    #

    def run_model(self, params):

        NOT_USED_NUM = 0.5

        sevs_trait_off = np.asarray(self.input_severities)

        output_trait_on = (
            Simulator(
                host_plant_on=False,
                fungicide_on=True,
                number_of_sprays=1,
                l_mu=NOT_USED_NUM,
                l_b=NOT_USED_NUM,
                k_mu=params['mu'],
                k_b=params['b'],
                config=self.config
            )
            .run_model(
                I0_vec=self.I0s,
                beta_vec=self.betas
            )
        )

        sevs_trait_on = np.asarray(output_trait_on['dis_sev'])

        control = 100 * (1 - sevs_trait_on / sevs_trait_off)

        return control

    #
    #

    def get_params(self, trial):
        params = {
            # GAMMA
            "mu": trial.suggest_float("mu", 1e-2, 40),
            "b": trial.suggest_float("b", 1e-3, 20, log=True),

            # BETA
            # "mu": trial.suggest_float("mu", 1e-6, 1, log=True),
            # "b": trial.suggest_float("b", 1e-4, 30, log=True),
        }
        return params


#
#
#
#
#


class HostMaxMutationObjective:
    def __init__(self, config) -> None:
        """Fit maximum mutation scale for host trait

        Parameters
        ----------
        config : Config
            See Config docs - type 'single'

        Example
        -------
        >>>sampler = TPESampler(seed=0)
        >>>study = optuna.create_study(sampler=sampler)
        >>>obj_h = HostMaxMutationObjective(host_fit_config)

        """

        self.df = (
            pd.read_csv(
                '../data/03_model_inputs/control_host.csv'
            )
            .iloc[:, 1:]
            .loc[lambda df: df.cultivar == config.cultivar]
            .loc[lambda df: df.year.isin([df.year.min(), df.year.max()])]

            # FOR MUTATION, KEEPING FULL DF

            # .loc[lambda df: (
            #     df.year <= (
            #         df.year.min()
            #         + train_test_split_proportion *
            #         (df.year.max() - df.year.min())
            #     )
            # )
            # ]

            .rename(columns={
                'control': 'data_control',
                # 'min_num': 'n_data',
            })

            # NB n_data = 4 for all points in year 0, 9 so this is ok
            .groupby('year').mean()

            .reset_index()
        )

        input_severities = (
            pd.read_csv(
                '../data/03_model_inputs/input_severities_host.csv'
            )
            .iloc[:, 1:]
            .loc[lambda df: df.cultivar == config.cultivar]
        )

        self.input_severities = 0.01*np.array(input_severities.worst_stb)

        n_years = len(input_severities)

        self.I0s = [config.I0_single] * n_years

        self.betas = find_beta_vectorised(
            0.01*input_severities.worst_stb,
            # np.array([0.01] * n_years),
            config.I0_single
        )

        self.config = config

    #
    #

    def __call__(self, trial):
        if trial is None:
            params = {'mean': 8, 'mutation_scale': 0.01}
        else:
            params = self.get_params(trial)

        control = self.run_model(params)

        return score_for_this_df(self.df, control)

    #
    #

    def run_model(self, params):

        config_use = copy.deepcopy(self.config)

        config_use.mutation_scale_host = params['mutation_scale']

        NOT_USED_NUM = 0.5

        point_dist = initial_point_distribution(
            config_use.n_l,
            params['mean']
        )

        sevs_trait_off = self.input_severities

        sim_on = Simulator(
            host_plant_on=True,
            fungicide_on=False,
            l_mu=NOT_USED_NUM,
            l_b=NOT_USED_NUM,
            k_mu=NOT_USED_NUM,
            k_b=NOT_USED_NUM,
            config=config_use
        )

        sim_on.initial_l_dist = point_dist

        output_trait_on = (
            sim_on
            .run_model(
                I0_vec=self.I0s,
                beta_vec=self.betas
            )
        )

        sevs_trait_on = np.asarray(output_trait_on['dis_sev'])

        control = 100 * (1 - sevs_trait_on / sevs_trait_off)

        return control

    #
    #

    def get_params(self, trial):
        params = {
            "mean": trial.suggest_float(
                "mean",
                1e-2,
                #
                #
                # make sure starts high? >
                # 0.85,
                1,
            ),
            "mutation_scale": trial.suggest_float(
                "mutation_scale",
                1e-4,
                20,
                log=True),
        }
        return params


#
#
#
#
#


class FungMaxMutationObjective:
    def __init__(self, config) -> None:
        """Fit maximum mutation scale for fungicide trait

        Parameters
        ----------
        config : Config
            See Config docs - type 'single'

        Example
        -------
        >>>sampler = TPESampler(seed=0)
        >>>study = optuna.create_study(sampler=sampler)
        >>>obj_h = FungMaxMutationObjective(fung_fit_config)

        """

        raw_df = pd.read_csv('../data/03_model_inputs/control_prothio.csv')

        self.df = (
            raw_df

            # FOR MUTATION, KEEPING FULL DF

            # .loc[lambda df: (
            #     df.year <= (
            #         df.year.min()
            #         + train_test_split_proportion *
            #         (df.year.max() - df.year.min())
            #     )
            # )
            # ]

            .loc[lambda df: df.year.isin([df.year.min(), df.year.max()])]

            .rename(columns={
                'control': 'data_control',
                # 'min_num': 'n_data',
            })
        )

        self.input_severities = [0.37] * raw_df.shape[0]

        self.I0s = [config.I0_single] * raw_df.shape[0]

        self.betas = find_beta_vectorised(
            self.input_severities, config.I0_single)

        self.config = config

    #
    #

    def __call__(self, trial):
        if trial is None:
            params = {'mean': 8, 'mutation_scale': 0.01}
        else:
            params = self.get_params(trial)

        control = self.run_model(params)

        return score_for_this_df(self.df, control)

    #
    #

    def run_model(self, params):

        config_use = copy.deepcopy(self.config)

        config_use.mutation_scale_fung = params['mutation_scale']

        NOT_USED_NUM = 0.5

        point_dist = initial_point_distribution(
            config_use.n_k,
            params['mean']
        )

        sevs_trait_off = np.asarray(self.input_severities)

        sim_on = Simulator(
            host_plant_on=False,
            fungicide_on=True,
            number_of_sprays=1,
            l_mu=NOT_USED_NUM,
            l_b=NOT_USED_NUM,
            k_mu=NOT_USED_NUM,
            k_b=NOT_USED_NUM,
            config=config_use
        )

        sim_on.initial_k_dist = point_dist

        output_trait_on = (
            sim_on
            .run_model(
                I0_vec=self.I0s,
                beta_vec=self.betas
            )
        )

        sevs_trait_on = np.asarray(output_trait_on['dis_sev'])

        control = 100 * (1 - sevs_trait_on / sevs_trait_off)

        return control

    #
    #

    def get_params(self, trial):
        params = {
            "mean": trial.suggest_float(
                "mean",
                1e-3,
                1,
                log=True
            ),
            "mutation_scale": trial.suggest_float(
                "mutation_scale",
                1e-4,
                5,
                log=True
            ),
        }
        return params


def fitting_df(config, study, trait=None):
    """Get dataframe from config and study to add to list of fitted values

    Parameters
    ----------
    config : Config
        _description_
    study : Optuna study
        _description_
    trait : str, optional
        - if host, leave as None - will find which cultivar from config;
        - if fung, set to 'Fungicide';
        by default None

    Returns
    -------
    pd.DataFrame
    """
    time = datetime.now().strftime('%Y-%m-%d %H:%M')

    if trait is None:
        # FITTING CULTIVAR
        trait = config.cultivar
        shape_par = study.best_params['b']
    else:
        # FITTING FUNG
        shape_par = study.best_params['b']

    return pd.DataFrame([
        dict(
            trait=trait,
            mutation_prop=config.mutation_proportion,
            mutation_scale_host=config.mutation_scale_host,
            mutation_scale_fung=config.mutation_scale_fung,
            mu=study.best_params['mu'],
            b=shape_par,
            score=int(study.best_trial.values[0]),
            nk=config.n_k,
            nl=config.n_l,
            trial_number=study.best_trial.number,
            date=time,
        )
    ])
