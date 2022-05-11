from math import floor

import numpy as np
from scipy import signal
from scipy.integrate import ode
from scipy.stats import norm

from poly2.params import PARAMS
from poly2.utils import (
    Fungicide,
    economic_yield_function,
    get_host_dist_params,
    get_fung_dist_params,
    trait_vec,
    edge_values,
    host_growth_function,
    initial_host_distribution,
    initial_fung_distribution,
    normalise,
    yield_function
)

#
#
#


class Simulator:
    """Sets up and runs a single model simulation. 

    Called by RunModel and ModelFitting.
    """

    def __init__(
        self,
        config,
        k_mu=None,
        k_b=None,
        l_mu=None,
        l_b=None,
        host_plant_on=True,
        fungicide_on=True,
        number_of_sprays=0,
        custom_sprays_vec=None,
    ):
        """Init method

        Parameters
        ----------
        config : Config
            See docs for Config

        k_mu : float, optional
            Fungicide distribution parameter.
            Overrides config k_mu, by default None

        k_b : float, optional
            Fungicide distribution parameter.
            Overrides config k_b, by default None

        l_mu : float, optional
            Host plant distribution parameter.
            Overrides config l_mu, by default None

        l_b : float, optional
            Host plant distribution parameter.
            Overrides config l_b, by default None

        host_plant_on : bool, optional
            Whether host plant protection is on, by default True
            Used instead of config, since might be iterating over multiple values
            a list in the config.

        fungicide_on : bool, optional
            Whether fungicide protection is on, by default True.
            Used instead of config, since might be iterating over multiple values
            a list in the config.

        number_of_sprays : int, optional
            N sprays per year, overridden by custom sprays vec, by default 0

        custom_sprays_vec : list/np.array, optional
            Overrides number of sprays, by default None
        """

        self.host_plant_on = host_plant_on
        self.fungicide_on = fungicide_on

        self.number_of_sprays = number_of_sprays
        self.custom_sprays_vec = custom_sprays_vec

        self.config = config

        self.k_a, self.k_b = get_fung_dist_params(k_mu, k_b, self.config)

        self.l_a, self.l_b = get_host_dist_params(l_mu, l_b, self.config)

        self.n_k = self.config.n_k
        self.n_l = self.config.n_l

        self.k_vec = trait_vec(self.n_k)
        self.l_vec = trait_vec(self.n_l)

        self.mutation_array = None
        self.fung_kernel = None
        self.host_kernel = None

        self.initial_l_dist = initial_host_distribution(
            self.n_l,
            self.l_a,
            self.l_b
        )

        self.initial_k_dist = initial_fung_distribution(
            self.n_k,
            self.k_a,
            self.k_b
        )

        #
        #

    def run_model(self, I0_vec, beta_vec):
        """_summary_

        Parameters
        ----------
        I0_vec : np.array/list
            A single I0 value per year
        beta_vec : np.array/list
            A single beta value per year

        Returns
        -------
        dict
            keys:
            - fung_dists: np.array, shape (n_k, n_years+1) - includes year 0
            - host_dists: np.array, shape (n_l, n_years+1)

            - n_k: int
            - n_l: int

            - I0_vec: np.array, shape (n_years, )
            - beta_vec: np.array, shape (n_years, )

            - t: np.array, shape (n_timepoints, )
            - y: np.array, shape (1+ [n_k OR n_l OR n_k*n_l], n_t, n_years)

            - total_I: np.array, shape (n_timepoints, n_years)

            - dis_sev: np.array, shape (n_years, )
            - yield_vec: np.array, shape (n_years, )
            - econ: np.array, shape (n_years, )
        """

        replace_cultivar_array = self.config.replace_cultivars

        fung_dists = np.zeros((self.n_k, len(beta_vec)+1))
        host_dists = np.zeros((self.n_l, len(beta_vec)+1))

        fung_dists[:, 0] = self.initial_k_dist
        host_dists[:, 0] = self.initial_l_dist

        dis_sev = np.zeros(len(beta_vec))

        if self.custom_sprays_vec is not None:
            sprays_vec_use = self.custom_sprays_vec
        else:
            sprays_vec_use = self.number_of_sprays*np.ones(len(beta_vec))

        #
        # run 0th year
        (
            sol, t, fung_dists[:, 1], host_dists[:, 1], total_infection
        ) = self.calculate_ode_soln(
            fung_dists[:, 0],
            host_dists[:, 0],
            I0_vec[0],
            beta_vec[0],
            sprays_vec_use[0]
        )

        # get shape of solution arrays from the output
        sol_list = np.zeros((sol.shape[0], sol.shape[1], len(beta_vec)))
        total_I = np.zeros((len(t), len(beta_vec)))

        # set the first year
        sol_list[:, :, 0] = sol
        total_I[:, 0] = total_infection
        dis_sev[0] = total_I[-1, 0]

        if self.config.host_growth:
            # scale so proportion of final leaf size
            S_end = sol_list[-1, -1, 0]
            dis_sev[0] = dis_sev[0] / (dis_sev[0] + S_end)

        if replace_cultivar_array is not None and replace_cultivar_array[0]:
            host_dists[:, 1] = self.initial_l_dist

        #
        # calculate the rest of the years
        for yr in range(1, len(beta_vec)):

            (
                sol_list[:, :, yr], t, fung_dists[:, yr+1],
                host_dists[:, yr+1], total_I[:, yr]
            ) = self.calculate_ode_soln(
                fung_dists[:, yr],
                host_dists[:, yr],
                I0_vec[yr],
                beta_vec[yr],
                sprays_vec_use[yr]
            )

            dis_sev[yr] = total_I[-1, yr]

            # scale so proportion of final leaf size
            S_end = sol_list[-1, -1, yr]
            dis_sev[yr] = dis_sev[yr] / (dis_sev[yr] + S_end)

            if (replace_cultivar_array is not None
                    and replace_cultivar_array[yr]):
                host_dists[:, yr+1] = self.initial_l_dist

        # calculate yield and economic yield
        yield_vec = [yield_function(sev) for sev in dis_sev]

        econ = [
            economic_yield_function([yield_], int(spray_num))[0]
            for yield_, spray_num in zip(yield_vec, sprays_vec_use)
        ]

        return {
            'fung_dists': fung_dists,
            'host_dists': host_dists,
            'n_k': self.n_k,
            'n_l': self.n_l,
            'k_vec': self.k_vec,
            'l_vec': self.l_vec,
            'I0_vec': I0_vec,
            'beta_vec': beta_vec,
            't': t,
            'y': sol_list,
            'total_I': total_I,
            'dis_sev': dis_sev,
            'yield_vec': np.asarray(yield_vec),
            'econ': np.asarray(econ)
        }

    def calculate_ode_soln(self, D0_k, D0_l, I0_in, beta_in, num_sprays):

        # self._reduce_box_numbers(D0_k, D0_l, will_reduce)

        self._get_y0(I0_in, D0_l, D0_k)

        solution_tmp, solutiont = self._solve_it(beta_in, num_sprays)

        solution, fung_dist_out, host_dist_out = self._generate_new_dists(
            solution_tmp, D0_l, D0_k)

        total_infection = [sum(solution[:-1, tt])
                           for tt in range(len(solutiont))]
        total_infection = np.asarray(total_infection)

        return solution, solutiont, fung_dist_out, host_dist_out, total_infection

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

        old_I = y[:-1]
        new_I = np.matmul(mutation_array, old_I)

        # host effect is same as value of strain
        host_effect_vec = np.array(strains_dict['host'])

        # fung effect is value of strain into fungicide effect function,
        # which depends on dose
        fung_effect_vec = np.array([
            fungicide.effect(strain, t) for strain in strains_dict['fung']
        ])

        disease_states = host_effect_vec * fung_effect_vec * new_I

        dydt[:-1] = beta * S * disease_states

        dydt[-1] = host_growth_fn(t, S, y) - beta * S * sum(disease_states)

        return dydt

    def _get_y0(self, I0_in, D0_l, D0_k):
        """
        y0 different if we have two active traits vs one
        """

        S0_prop = 1 - I0_in

        # set initial condition
        if self.host_plant_on and self.fungicide_on:
            y0_use = np.zeros(self.n_k*self.n_l+1)

            infection_array = (
                I0_in * np.outer(D0_k, D0_l)
            )

            infection_vector = np.reshape(
                infection_array,
                (self.n_k*self.n_l)
            )

            y0_use[:-1] = infection_vector

        elif self.fungicide_on and not self.host_plant_on:
            y0_use = np.zeros(self.n_k+1)
            y0_use[:-1] = I0_in*D0_k

        elif self.host_plant_on and not self.fungicide_on:
            y0_use = np.zeros(self.n_l+1)
            y0_use[:-1] = I0_in*D0_l

        y0_use[-1] = S0_prop

        self.y0 = PARAMS.host_growth_initial_area*y0_use

    def _get_mutation_array(self):

        if self.fung_kernel is None:
            self.fung_kernel = self._get_kernel(
                self.k_vec,
                self.config.mutation_proportion,
                self.config.mutation_scale_fung,
            )

        if self.host_kernel is None:
            self.host_kernel = self._get_kernel(
                self.l_vec,
                self.config.mutation_proportion,
                self.config.mutation_scale_host,
            )

        if self.fungicide_on and self.host_plant_on:
            mutate_array = np.zeros(
                (self.n_k*self.n_l,
                 self.n_k*self.n_l)
            )

            fung_ker = self.fung_kernel
            host_ker = self.host_kernel

            for i in range(mutate_array.shape[0]):
                for j in range(mutate_array.shape[1]):
                    # convert to "fung index" in [0,self.n_k-1]
                    f1 = floor(i/self.n_l)
                    # convert to "fung index" in [0,self.n_k-1]
                    f2 = floor(j/self.n_l)

                    # convert to "host index" in [0,self.n_l-1]
                    h1 = i % self.n_l
                    # convert to "host index" in [0,self.n_l-1]
                    h2 = j % self.n_l

                    mutate_array[i, j] = fung_ker[f1, f2] * host_ker[h1, h2]

            self.mutation_array = mutate_array

        elif self.fungicide_on and not self.host_plant_on:
            self.mutation_array = self.fung_kernel

        elif not self.fungicide_on and self.host_plant_on:
            self.mutation_array = self.host_kernel

    def _get_kernel(self, vec, p, mutation_scale):

        N = len(vec)

        kernel = np.zeros((N, N))

        for parent in range(N):
            # some proportion stays at position i
            not_dispersing = signal.unit_impulse(N, parent)

            dispersing = self.dispersal(vec, parent, mutation_scale)

            kernel[parent, :] = p*dispersing + (1-p)*not_dispersing

        return kernel

    def dispersal(self, vec, parent_index, mut_scale):

        stan_dev = mut_scale**0.5

        edges = edge_values(len(vec))

        disp = norm.cdf(edges, loc=vec[parent_index], scale=stan_dev)

        dispersing = np.diff(disp)

        top = 1 - disp[-1]

        bottom = disp[0]

        dispersing[0] += bottom

        dispersing[-1] += top

        return dispersing

    def _solve_it(self, beta_in, num_sprays):

        strains_dict = {}

        trait_vec_dict = {
            'host': np.asarray(self.l_vec),
            'fung': np.asarray(self.k_vec),
        }

        ode_solver = ode(self._ode_system_with_mutation)
        self._get_mutation_array()

        ode_solver.set_integrator('dopri5', max_step=10)

        t_out = np.linspace(PARAMS.T_1, PARAMS.T_end, 100)
        t_out = list(t_out)

        # += PARAMS.T_1, no longer needed!
        t_out += [PARAMS.T_2, PARAMS.T_3]
        t_out = sorted(t_out)
        t_out = np.asarray(t_out)

        y_out = np.zeros((self.y0.shape[0], len(t_out)))

        ode_solver.set_initial_value(self.y0, t_out[0])

        if self.fungicide_on and self.host_plant_on:

            for trait in ['host', 'fung']:

                if trait == 'host':
                    vec1 = np.ones(len(trait_vec_dict['fung']))
                    vec2 = trait_vec_dict['host']
                else:
                    vec1 = trait_vec_dict['fung']
                    vec2 = np.ones(len(trait_vec_dict['host']))

                # distribute these according to the length of the reshaped vector
                trait_effect_array = np.outer(vec1, vec2)

                strains_dict[trait] = np.reshape(
                    trait_effect_array,
                    y_out.shape[0]-1,
                    order='C'
                )

        elif self.fungicide_on and not self.host_plant_on:
            # NB looking to match length of active fung trait vector, not host
            strains_dict['host'] = np.ones(len(trait_vec_dict['fung']))
            strains_dict['fung'] = trait_vec_dict['fung']

        elif not self.fungicide_on and self.host_plant_on:
            # NB looking to match length of active host trait vector, not fung
            strains_dict['host'] = trait_vec_dict['host']
            strains_dict['fung'] = np.ones(len(trait_vec_dict['host']))

        # add other params
        my_fungicide = Fungicide(num_sprays)

        host_growth_fn = host_growth_function

        ode_solver.set_f_params(
            beta_in,
            host_growth_fn,
            strains_dict,
            my_fungicide,
            self.mutation_array
        )

        for ind, tt in enumerate(t_out[1:]):
            if ode_solver.successful():
                y_out[:, ind] = ode_solver.y
                ode_solver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')

        y_out[:, -1] = ode_solver.y

        return y_out, t_out

    def _generate_new_dists(self, solution, D0_l, D0_k):

        I_end = normalise(solution[:-1, -1])

        n_t_points = solution.shape[1]

        if self.host_plant_on and self.fungicide_on:
            soln_out, fung_dist_out, host_dist_out = self._dists_fung_and_host(
                solution)

        elif self.fungicide_on and not self.host_plant_on:
            soln_out = np.zeros((self.n_k+1, n_t_points))
            soln_out[self.k_start:-1, :] = solution[:-1, :]

            fung_dist_out = np.zeros(self.n_k)
            fung_dist_out = I_end
            host_dist_out = D0_l

        elif self.host_plant_on and not self.fungicide_on:
            soln_out = np.zeros((self.n_l+1, n_t_points))
            soln_out[:-1, :] = solution[:-1, :]

            host_dist_out = np.zeros(self.n_l)
            host_dist_out = I_end
            fung_dist_out = D0_k

        # susceptible tissue
        soln_out[-1, :] = solution[-1, :]

        return soln_out, fung_dist_out, host_dist_out

    def _dists_fung_and_host(self, solution):
        I_end = solution[:-1, -1]

        n_t_points = solution.shape[1]

        I0_k_end = np.zeros(self.n_k)
        I0_l_end = np.zeros(self.n_l)

        I_end_reshaped = np.reshape(I_end, (self.n_k, self.n_l))

        I0_k_end = np.asarray(
            [sum(I_end_reshaped[k, :]) for k in range(I_end_reshaped.shape[0])]
        )

        I0_l_end = np.asarray(
            [sum(I_end_reshaped[:, l]) for l in range(I_end_reshaped.shape[1])]
        )

        fung_dist_out = normalise(I0_k_end)
        host_dist_out = normalise(I0_l_end)

        soln_large_array = np.zeros(((self.n_k, self.n_l, n_t_points)))

        soln_small_array = np.reshape(
            solution[:-1, :],
            (I_end_reshaped.shape[0], I_end_reshaped.shape[1], n_t_points),
        )

        soln_large_array = soln_small_array

        solution_out = np.zeros((self.n_k*self.n_l+1, n_t_points))

        solution_out[:-1, :] = np.reshape(
            soln_large_array,
            (self.n_k * self.n_l, n_t_points),
        )

        return solution_out, fung_dist_out, host_dist_out
