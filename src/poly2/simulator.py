"""Polygenic models in here

Various different types of simulator
- SimulatorOneTrait
- SimulatorAsymptote

Older:
- SimulatorSimple (only fung trait, no host growth/density dependance/
    mutation/fung decay)
- SimulatorSimpleWithDD
"""

import numpy as np
from scipy.integrate import ode

from poly2.params import PARAMS
from poly2.utils import (
    Fungicide,
    FungicideAsymptote,
    FungicideNoDecay,
    disease_severity,
    get_dist_mean,
    yield_fn,
    economic_yield,
    host_growth_function,
    trait_vec,
    beta_dist,
    gamma_dist,
    get_fung_dist_params_from_config,
    get_host_dist_params_from_config,
    normalise,
    get_dispersal_kernel,
    get_model_times,
)


class SimulatorOneTrait:
    """Sets up and runs a single model simulation for fung OR host trait.

    Example
    -------
    >>>cf = Config()
    >>># or e.g. Config(sprays=[2], host_on=[False])
    >>>data = SimulatorOneTrait(cf).run_model()
    """

    def __init__(
        self,
        config,
        fungicide_on=True,
        host_plant_on=False,
        number_of_sprays=2,
    ):
        """Init method

        Parameters
        ----------
        config : Config
            See docs for Config

        host_plant_on : bool, optional
            Whether host plant protection is on, by default True
            Used instead of config, since might be iterating over multiple values
            a list in the config.

        fungicide_on : bool, optional
            Whether fungicide protection is on, by default True.
            Used instead of config, since might be iterating over multiple values
            a list in the config.

        number_of_sprays : int, optional
            N sprays per year, by default 0
        """

        self.host_plant_on = host_plant_on
        self.fungicide_on = fungicide_on

        if host_plant_on:
            self.number_of_sprays = 0
        else:
            self.number_of_sprays = number_of_sprays

        self.conf_o = config
        self.mutation_array = None
        self.y0 = None

        self.k_a, self.k_b = get_fung_dist_params_from_config(self.conf_o)
        self.l_a, self.l_b = get_host_dist_params_from_config(self.conf_o)

        self.n_k = self.conf_o.n_k
        self.n_l = self.conf_o.n_l

        self.n_years = self.conf_o.n_years

        self.k_vec = trait_vec(self.n_k)
        self.l_vec = trait_vec(self.n_l)

        self.t = get_model_times()

        self.initial_k_dist = gamma_dist(self.n_k, self.k_a, self.k_b)
        self.initial_l_dist = beta_dist(self.n_l, self.l_a, self.l_b)

        #
        #

    def run_model(self):
        """_summary_

        Returns
        -------
        dict
            keys:
            - fung_dists: np.array, shape (n_k, n_years+1) - includes year 0
            - host_dists: np.array, shape (n_l, n_years+1)

            - fung_mean: np.array, shape (n_years+1,) - includes year 0
            - host_mean: np.array, shape (n_years+1,) - includes year 0

            - n_k: int
            - n_l: int

            - k_vec: np.array, shape (n_k, )
            - l_vec: np.array, shape (n_l, )

            - I0s: np.array, shape (n_years, )
            - betas: np.array, shape (n_years, )

            - t: np.array, shape (n_timepoints, )
            - y: np.array, shape (1+ [n_k OR n_l], n_t, n_years)

            - total_I: np.array, shape (n_timepoints, n_years)

            - dis_sev: np.array, shape (n_years, )
            - yield_vec: np.array, shape (n_years, )
            - econ: np.array, shape (n_years, )
            - years: np.array, shape (n_years, )
        """

        replace_cultivar_array = self.conf_o.replace_cultivars

        fung_dists = np.zeros((self.n_k, self.n_years+1))
        host_dists = np.zeros((self.n_l, self.n_years+1))

        fung_dists[:, 0] = self.initial_k_dist
        host_dists[:, 0] = self.initial_l_dist

        dis_sev = np.zeros(self.n_years)

        sprays_vec = self.number_of_sprays*np.ones(self.n_years)

        self._get_mutation_kernels()

        if self.fungicide_on and not self.host_plant_on:
            y_len = self.n_k+1
        elif self.host_plant_on and not self.fungicide_on:
            y_len = self.n_l+1

        sol_arr = np.zeros((y_len, len(self.t), self.n_years))
        total_I = np.zeros((len(self.t), self.n_years))

        #
        for yr in range(self.n_years):

            (
                sol_arr[:, :, yr],
                fung_dists[:, yr+1],
                host_dists[:, yr+1],
                total_I[:, yr]
            ) = self.calculate_ode_soln(
                fung_dists[:, yr],
                host_dists[:, yr],
                self.conf_o.I0s[yr],
                self.conf_o.betas[yr],
                sprays_vec[yr],
                self.conf_o.doses[yr],
            )

            if (replace_cultivar_array is not None
                    and replace_cultivar_array[yr]):
                host_dists[:, yr+1] = self.initial_l_dist

        # calculate disease severity, yield and economic yield
        dis_sev = disease_severity(total_I[-1, :], sol_arr[-1, -1, :])

        yield_vec = np.array([yield_fn(sev) for sev in dis_sev])

        econ = economic_yield(
            yield_vec, sprays_vec, self.conf_o.doses)

        fung_mean = get_dist_mean(fung_dists, self.k_vec)
        host_mean = get_dist_mean(host_dists, self.l_vec)

        return {
            'fung_dists': fung_dists,
            'host_dists': host_dists,
            'fung_mean': fung_mean,
            'host_mean': host_mean,
            'n_k': self.n_k,
            'n_l': self.n_l,
            'k_vec': self.k_vec,
            'l_vec': self.l_vec,
            'I0s': self.conf_o.I0s,
            'betas': self.conf_o.betas,
            't': self.t,
            'y': sol_arr,
            'total_I': total_I,
            'dis_sev': dis_sev,
            'yield_vec': yield_vec,
            'econ': econ,
            'year': np.arange(1, 1 + len(yield_vec))
        }

    def calculate_ode_soln(self, D0_k, D0_l, I0_in, beta_in, num_sprays, dose):

        self._get_y0(I0_in, D0_l, D0_k)

        soln = self._solve_it(beta_in, num_sprays, dose)

        fung_dist_out, host_dist_out = self._generate_new_dists(
            soln, D0_l, D0_k
        )

        total_infection = soln[:-1, :].sum(axis=0)

        return soln, fung_dist_out, host_dist_out, total_infection

    def _ode_system_with_mutation(
            self,
            t,
            y,
            beta,
            host_growth_fn,
            strains_dict,
            fungicide,
            mutation_array
    ):

        dydt = np.zeros(len(self.y0))

        S = y[-1]

        I_in = y[:-1]

        # host effect is value of strain (host on) or np.ones (host off)
        host_effect_vec = np.array(strains_dict['host'])

        # fung effect is value of strain into fungicide effect function
        fung_effect_vec = np.array([
            fungicide.effect(strain, t) for strain in strains_dict['fung']
        ])

        offspring = host_effect_vec * fung_effect_vec * I_in

        I_after_mutation = np.matmul(mutation_array, offspring)

        dydt[:-1] = beta * S * I_after_mutation

        dydt[-1] = host_growth_fn(t, S, y) - beta * S * sum(I_after_mutation)

        return dydt

    def _get_y0(self, I0_in, D0_l, D0_k):
        """
        y0 different if we have two active traits vs one
        """

        S0_prop = 1 - I0_in

        # set initial condition
        if self.fungicide_on and not self.host_plant_on:
            y0_use = np.zeros(self.n_k+1)
            y0_use[:-1] = I0_in*D0_k

        elif self.host_plant_on and not self.fungicide_on:
            y0_use = np.zeros(self.n_l+1)
            y0_use[:-1] = I0_in*D0_l

        y0_use[-1] = S0_prop

        self.y0 = PARAMS.host_growth_initial_area*y0_use

    def _get_mutation_kernels(self):

        if self.fungicide_on and not self.host_plant_on:
            # FUNG
            self.mutation_array = get_dispersal_kernel(
                self.k_vec,
                self.conf_o.mutation_proportion,
                self.conf_o.mutation_scale_fung,
            )

        elif not self.fungicide_on and self.host_plant_on:
            # HOST
            self.mutation_array = get_dispersal_kernel(
                self.l_vec,
                self.conf_o.mutation_proportion,
                self.conf_o.mutation_scale_host,
            )

    def _solve_it(self, beta_in, num_sprays, dose):

        ode_solver = ode(self._ode_system_with_mutation)

        ode_solver.set_integrator('dopri5', max_step=10)

        y_out = np.zeros((self.y0.shape[0], len(self.t)))

        ode_solver.set_initial_value(self.y0, self.t[0])

        strains_dict = {}
        if self.fungicide_on and not self.host_plant_on:
            # NB looking to match length of active fung trait vector, not host
            strains_dict['host'] = np.ones(self.n_k)
            strains_dict['fung'] = np.array(self.k_vec)

        elif not self.fungicide_on and self.host_plant_on:
            # NB looking to match length of active host trait vector, not fung
            strains_dict['host'] = np.array(self.l_vec)
            strains_dict['fung'] = np.ones(self.n_l)

        # add other params
        my_fungicide = Fungicide(
            num_sprays,
            dose,
            self.conf_o.decay_rate,
            self.conf_o.asymptote,
        )

        ode_solver.set_f_params(
            beta_in,
            host_growth_function,
            strains_dict,
            my_fungicide,
            self.mutation_array
        )

        for ind, tt in enumerate(self.t[1:]):
            if ode_solver.successful():
                y_out[:, ind] = ode_solver.y
                ode_solver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')

        y_out[:, -1] = ode_solver.y

        return y_out

    def _generate_new_dists(self, solution, D0_l, D0_k):

        I_end = normalise(solution[:-1, -1])

        if self.fungicide_on and not self.host_plant_on:
            fung_dist_out = I_end
            host_dist_out = D0_l

        elif self.host_plant_on and not self.fungicide_on:
            host_dist_out = I_end
            fung_dist_out = D0_k

        return fung_dist_out, host_dist_out


#
#
#

class SimulatorAsymptote:
    """Sets up and runs a single model simulation for fung trait with asymptote
      instead of curvature as the fungicide trait.

      Example
      -------
      >>>cf = Config()
      >>># or e.g. Config(sprays=[2], host_on=[False])
      >>>data = SimulatorOneTrait(cf).run_model()
      """

    def __init__(
        self,
        config,
        number_of_sprays=2,
    ):
        """Init method

        Parameters
        ----------
        config : Config
            See docs for Config

        number_of_sprays : int, optional
            N sprays per year, by default 0
        """

        self.number_of_sprays = number_of_sprays

        self.conf_a = config
        self.mutation_array = None
        self.y0 = None

        kmu = config.k_mu

        self.k_b = config.k_b
        self.k_a = (self.k_b*kmu)/(1-kmu)

        self.n_k = self.conf_a.n_k

        self.n_years = self.conf_a.n_years

        self.k_vec = trait_vec(self.n_k)

        self.t = get_model_times()

        self.initial_k_dist = beta_dist(self.n_k, self.k_a, self.k_b)

        #
        #

    def run_model(self):
        """_summary_

        Returns
        -------
        dict
            keys:
            - fung_dists: np.array, shape (n_k, n_years+1) - includes year 0

            - fung_mean: np.array, shape (n_years+1,) - includes year 0

            - n_k: int

            - k_vec: np.array, shape (n_k, )

            - I0s: np.array, shape (n_years, )
            - betas: np.array, shape (n_years, )

            - t: np.array, shape (n_timepoints, )
            - y: np.array, shape (1+ [n_k OR n_l], n_t, n_years)

            - total_I: np.array, shape (n_timepoints, n_years)

            - dis_sev: np.array, shape (n_years, )
            - yield_vec: np.array, shape (n_years, )
            - econ: np.array, shape (n_years, )
            - years: np.array, shape (n_years, )
        """

        fung_dists = np.zeros((self.n_k, self.n_years+1))

        fung_dists[:, 0] = self.initial_k_dist

        dis_sev = np.zeros(self.n_years)

        sprays_vec = self.number_of_sprays*np.ones(self.n_years)

        self.mutation_array = get_dispersal_kernel(
            self.k_vec,
            self.conf_a.mutation_proportion,
            self.conf_a.mutation_scale_fung,
        )

        y_len = self.n_k+1

        sol_arr = np.zeros((y_len, len(self.t), self.n_years))
        total_I = np.zeros((len(self.t), self.n_years))

        #
        for yr in range(self.n_years):

            (
                sol_arr[:, :, yr],
                fung_dists[:, yr+1],
                total_I[:, yr]
            ) = self.calculate_ode_soln(
                fung_dists[:, yr],
                self.conf_a.I0s[yr],
                self.conf_a.betas[yr],
                sprays_vec[yr],
                self.conf_a.doses[yr],
            )

        # calculate disease severity, yield and economic yield
        dis_sev = disease_severity(total_I[-1, :], sol_arr[-1, -1, :])

        yield_vec = np.array([yield_fn(sev) for sev in dis_sev])

        econ = economic_yield(yield_vec, sprays_vec, self.conf_a.doses)

        fung_mean = get_dist_mean(fung_dists, self.k_vec)

        return {
            'fung_dists': fung_dists,
            'fung_mean': fung_mean,
            'n_k': self.n_k,
            'k_vec': self.k_vec,
            'I0s': self.conf_a.I0s,
            'betas': self.conf_a.betas,
            't': self.t,
            'y': sol_arr,
            'total_I': total_I,
            'dis_sev': dis_sev,
            'yield_vec': yield_vec,
            'econ': econ,
            'year': np.arange(1, 1 + len(yield_vec))
        }

    def calculate_ode_soln(self, D0_k, I0_in, beta_in, num_sprays, dose):

        self._get_y0(I0_in, D0_k)

        soln = self._solve_it(beta_in, num_sprays, dose)

        fung_dist_out = self._generate_new_dists(soln)

        total_infection = soln[:-1, :].sum(axis=0)

        return soln, fung_dist_out, total_infection

    def _ode_system_with_mutation(
            self,
            t,
            y,
            beta,
            host_growth_fn,
            fungicide,
            mutation_array
    ):

        dydt = np.zeros(len(self.y0))

        S = y[-1]

        I_in = y[:-1]

        # fung effect is value of strain into fungicide effect function
        fung_effect_vec = np.array([
            fungicide.effect(strain, t) for strain in self.k_vec
        ])

        offspring = fung_effect_vec * I_in

        I_after_mutation = np.matmul(mutation_array, offspring)

        dydt[:-1] = beta * S * I_after_mutation

        dydt[-1] = host_growth_fn(t, S, y) - beta * S * sum(I_after_mutation)

        return dydt

    def _get_y0(self, I0_in, D0_k):
        """
        y0 different if we have two active traits vs one
        """

        S0_prop = 1 - I0_in

        # set initial condition
        y0_use = np.zeros(self.n_k+1)
        y0_use[:-1] = I0_in*D0_k

        y0_use[-1] = S0_prop

        self.y0 = PARAMS.host_growth_initial_area*y0_use

    def _solve_it(self, beta_in, num_sprays, dose):

        ode_solver = ode(self._ode_system_with_mutation)

        ode_solver.set_integrator('dopri5', max_step=10)

        y_out = np.zeros((self.y0.shape[0], len(self.t)))

        ode_solver.set_initial_value(self.y0, self.t[0])

        # add other params
        my_fungicide = FungicideAsymptote(
            num_sprays,
            dose,
            self.conf_a.curvature,
            self.conf_a.decay_rate,
        )

        ode_solver.set_f_params(
            beta_in,
            host_growth_function,
            my_fungicide,
            self.mutation_array
        )

        for ind, tt in enumerate(self.t[1:]):
            if ode_solver.successful():
                y_out[:, ind] = ode_solver.y
                ode_solver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')

        y_out[:, -1] = ode_solver.y

        return y_out

    def _generate_new_dists(self, solution):

        I_end = normalise(solution[:-1, -1])

        fung_dist_out = I_end

        return fung_dist_out


#
#
#


class SimulatorSimple:
    """Sets up and runs a single model simulation for simplified model.

    Simplifications:
    - only fung trait
    - no host growth
    - no density dependance (S=1)
    - no mutation
    - no fungicide decay

    Example
    -------
    >>>cf = Config()
    >>># or e.g. Config(sprays=[2], host_on=[False])
    >>>data = SimulatorSimple(cf).run_model()
    """

    def __init__(
        self,
        config,
        number_of_sprays=2,
    ):
        """Init method

        Parameters
        ----------
        config : Config
            See docs for Config

        number_of_sprays : int, optional
            N sprays per year, by default 0
        """

        self.number_of_sprays = number_of_sprays

        self.conf_s = config

        self.k_a, self.k_b = get_fung_dist_params_from_config(self.conf_s)

        self.n_k = self.conf_s.n_k

        self.y0 = None

        self.n_years = self.conf_s.n_years

        self.k_vec = trait_vec(self.n_k)

        self.t = get_model_times()

        self.initial_k_dist = gamma_dist(self.n_k, self.k_a, self.k_b)

        #
        #

    def run_model(self):
        """_summary_

        Returns
        -------
        dict
            keys:
            - fung_dists: np.array, shape (n_k, n_years+1) - includes year 0

            - fung_mean: np.array, shape (n_years+1,) - includes year 0

            - n_k: int

            - k_vec: np.array, shape (n_k, )

            - I0s: np.array, shape (n_years, )
            - betas: np.array, shape (n_years, )

            - t: np.array, shape (n_timepoints, )
            - y: np.array, shape (1+ [n_k OR n_l], n_t, n_years)

            - total_I: np.array, shape (n_timepoints, n_years)

            - dis_sev: np.array, shape (n_years, )
            - yield_vec: np.array, shape (n_years, )
            - econ: np.array, shape (n_years, )
            - years: np.array, shape (n_years, )
        """

        fung_dists = np.zeros((self.n_k, self.n_years+1))

        fung_dists[:, 0] = self.initial_k_dist

        dis_sev = np.zeros(self.n_years)

        sprays_vec = self.number_of_sprays*np.ones(self.n_years)

        y_len = self.n_k+1

        sol_arr = np.zeros((y_len, len(self.t), self.n_years))
        total_I = np.zeros((len(self.t), self.n_years))

        #
        for yr in range(self.n_years):

            (
                sol_arr[:, :, yr],
                fung_dists[:, yr+1],
                total_I[:, yr]
            ) = self.calculate_ode_soln(
                fung_dists[:, yr],
                self.conf_s.I0s[yr],
                self.conf_s.betas[yr],
                sprays_vec[yr],
                self.conf_s.doses[yr],
            )

        # calculate disease severity, yield and economic yield
        dis_sev = disease_severity(total_I[-1, :], sol_arr[-1, -1, :])

        yield_vec = np.array([yield_fn(sev) for sev in dis_sev])

        econ = economic_yield(
            yield_vec, sprays_vec, self.conf_s.doses)

        fung_mean = get_dist_mean(fung_dists, self.k_vec)

        return {
            'fung_dists': fung_dists,
            'fung_mean': fung_mean,
            'n_k': self.n_k,
            'k_vec': self.k_vec,
            'I0s': self.conf_s.I0s,
            'betas': self.conf_s.betas,
            't': self.t,
            'y': sol_arr,
            'total_I': total_I,
            'dis_sev': dis_sev,
            'yield_vec': yield_vec,
            'econ': econ,
            'year': np.arange(1, 1 + len(yield_vec))
        }

    def calculate_ode_soln(self, D0_k, I0_in, beta_in, num_sprays, dose):

        self._get_y0(I0_in, D0_k)

        soln = self._solve_it(beta_in, num_sprays, dose)

        fung_dist_out = normalise(soln[:-1, -1])

        total_infection = soln[:-1, :].sum(axis=0)

        return soln, fung_dist_out, total_infection

    def _ode_system_simple(
            self,
            t,
            y,
            beta,
            fungicide,
    ):

        dydt = np.zeros(len(self.y0))

        I_in = y[:-1]

        # fung effect is value of strain into fungicide effect function
        fung_effect_vec = np.array([
            fungicide.effect(strain, t) for strain in self.k_vec
        ])

        dydt[:-1] = beta * 1 * fung_effect_vec * I_in

        dydt[-1] = 0

        return dydt

    def _get_y0(self, I0_in, D0_k):
        """
        y0 different if we have two active traits vs one
        """

        # set initial condition
        y0_use = np.zeros(self.n_k+1)
        y0_use[:-1] = I0_in*D0_k

        y0_use[-1] = 1

        self.y0 = PARAMS.host_growth_initial_area*y0_use

    def _solve_it(self, beta_in, num_sprays, dose):

        ode_solver = ode(self._ode_system_simple)

        ode_solver.set_integrator('dopri5', max_step=10)

        y_out = np.zeros((self.y0.shape[0], len(self.t)))

        ode_solver.set_initial_value(self.y0, self.t[0])

        # add other params
        my_fungicide = FungicideNoDecay(num_sprays, dose)

        ode_solver.set_f_params(
            beta_in,
            my_fungicide,
        )

        for ind, tt in enumerate(self.t[1:]):
            if ode_solver.successful():
                y_out[:, ind] = ode_solver.y
                ode_solver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')

        y_out[:, -1] = ode_solver.y

        return y_out


class SimulatorSimpleWithDD:
    """Sets up and runs a single model simulation for simplified model.

    Simplifications:
    - only fung trait
    - no host growth
    - no mutation
    - no fungicide decay

    DOES have density dependence

    Example
    -------
    >>>cf = Config()
    >>># or e.g. Config(sprays=[2], host_on=[False])
    >>>data = SimulatorSimple(cf).run_model()
    """

    def __init__(
        self,
        config,
        number_of_sprays=2,
    ):
        """Init method

        Parameters
        ----------
        config : Config
            See docs for Config

        number_of_sprays : int, optional
            N sprays per year, by default 0
        """

        print('NB y0 diff to other simplified model')

        self.number_of_sprays = number_of_sprays

        self.conf_d = config

        self.k_a, self.k_b = get_fung_dist_params_from_config(self.conf_d)

        self.n_k = self.conf_d.n_k

        self.y0 = None

        self.n_years = self.conf_d.n_years

        self.k_vec = trait_vec(self.n_k)

        self.t = get_model_times()

        self.initial_k_dist = gamma_dist(self.n_k, self.k_a, self.k_b)

        #
        #

    def run_model(self):
        """_summary_

        Returns
        -------
        dict
            keys:
            - fung_dists: np.array, shape (n_k, n_years+1) - includes year 0

            - fung_mean: np.array, shape (n_years+1,) - includes year 0

            - n_k: int

            - k_vec: np.array, shape (n_k, )

            - I0s: np.array, shape (n_years, )
            - betas: np.array, shape (n_years, )

            - t: np.array, shape (n_timepoints, )
            - y: np.array, shape (1+ [n_k OR n_l], n_t, n_years)

            - total_I: np.array, shape (n_timepoints, n_years)

            - dis_sev: np.array, shape (n_years, )
            - yield_vec: np.array, shape (n_years, )
            - econ: np.array, shape (n_years, )
            - years: np.array, shape (n_years, )
        """

        fung_dists = np.zeros((self.n_k, self.n_years+1))

        fung_dists[:, 0] = self.initial_k_dist

        dis_sev = np.zeros(self.n_years)

        sprays_vec = self.number_of_sprays*np.ones(self.n_years)

        y_len = self.n_k+1

        sol_arr = np.zeros((y_len, len(self.t), self.n_years))
        total_I = np.zeros((len(self.t), self.n_years))

        #
        for yr in range(self.n_years):

            (
                sol_arr[:, :, yr],
                fung_dists[:, yr+1],
                total_I[:, yr]
            ) = self.calculate_ode_soln(
                fung_dists[:, yr],
                self.conf_d.I0s[yr],
                self.conf_d.betas[yr],
                sprays_vec[yr],
                self.conf_d.doses[yr],
            )

        # calculate disease severity, yield and economic yield
        dis_sev = disease_severity(total_I[-1, :], sol_arr[-1, -1, :])

        yield_vec = np.array([yield_fn(sev) for sev in dis_sev])

        econ = economic_yield(
            yield_vec, sprays_vec, self.conf_d.doses)

        fung_mean = get_dist_mean(fung_dists, self.k_vec)

        return {
            'fung_dists': fung_dists,
            'fung_mean': fung_mean,
            'n_k': self.n_k,
            'k_vec': self.k_vec,
            'I0s': self.conf_d.I0s,
            'betas': self.conf_d.betas,
            't': self.t,
            'y': sol_arr,
            'total_I': total_I,
            'dis_sev': dis_sev,
            'yield_vec': yield_vec,
            'econ': econ,
            'year': np.arange(1, 1 + len(yield_vec))
        }

    def calculate_ode_soln(self, D0_k, I0_in, beta_in, num_sprays, dose):

        self._get_y0(I0_in, D0_k)

        soln = self._solve_it(beta_in, num_sprays, dose)

        fung_dist_out = normalise(soln[:-1, -1])

        total_infection = soln[:-1, :].sum(axis=0)

        return soln, fung_dist_out, total_infection

    def _ode_system_simple(
            self,
            t,
            y,
            beta,
            fungicide,
    ):

        dydt = np.zeros(len(self.y0))

        S = y[-1]

        I_in = y[:-1]

        # fung effect is value of strain into fungicide effect function
        fung_effect_vec = np.array([
            fungicide.effect(strain, t) for strain in self.k_vec
        ])

        dis_states = fung_effect_vec * I_in

        dydt[:-1] = beta * S * dis_states

        dydt[-1] = - beta * S * sum(dis_states)

        return dydt

    def _get_y0(self, I0_in, D0_k):
        """
        y0 different if we have two active traits vs one

        NB have changed compared to others
        """

        # set initial condition
        y0_use = np.zeros(self.n_k+1)
        y0_use[:-1] = I0_in*D0_k

        y0_use[-1] = 1

        self.y0 = PARAMS.host_growth_initial_area*y0_use

    def _solve_it(self, beta_in, num_sprays, dose):

        ode_solver = ode(self._ode_system_simple)

        ode_solver.set_integrator('dopri5', max_step=10)

        y_out = np.zeros((self.y0.shape[0], len(self.t)))

        ode_solver.set_initial_value(self.y0, self.t[0])

        # add other params
        my_fungicide = FungicideNoDecay(num_sprays, dose)
        # my_fungicide = Fungicide(num_sprays, dose)

        ode_solver.set_f_params(
            beta_in,
            my_fungicide,
        )

        for ind, tt in enumerate(self.t[1:]):
            if ode_solver.successful():
                y_out[:, ind] = ode_solver.y
                ode_solver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')

        y_out[:, -1] = ode_solver.y

        return y_out
