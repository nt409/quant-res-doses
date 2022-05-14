"""Contains:

utility functions

classes:
Fungicide
"""

import itertools
import os
import pickle
from math import exp, log, log10

import numpy as np

from scipy.stats import beta, gamma
from scipy.optimize import minimize
from scipy.integrate import ode
from scipy import signal

from polymodel.consts import DEFAULT_I0, FUNG_DECAY_RATE
from polymodel.params import PARAMS


def normalise(dist):
    dist = np.asarray(dist)
    sum_of_dist = sum(dist)
    out = dist/sum_of_dist
    return out


def logit10(x):
    return log10(x/(1-x))


def logit10_vectorised(x):
    return np.vectorize(logit10)(x)


def inverse_logit10(x):
    return 10**(x) / (1 + 10**(x))


def object_dump(file_name, object_to_dump):
    """save object to pickle file"""

    # check if file path exists - if not create
    outdir = os.path.dirname(file_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(file_name, 'wb') as handle:
        pickle.dump(object_to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def find_beta_vectorised(final_sevs, I0):
    return np.vectorize(find_beta)(final_sevs, I0)


def ode_simple(t, y, beta_):
    dydt = np.zeros(2)

    S, I = y

    # dS
    dydt[0] = host_growth_function(t, S, y) - beta_ * S * I

    dydt[1] = beta_ * S * I

    return dydt


def find_soln_given_beta_and_no_control(beta_, I0=DEFAULT_I0, t_vals=None):
    y0 = np.array([
        PARAMS.host_growth_initial_area*(1-I0),
        PARAMS.host_growth_initial_area*I0
    ])

    solver = ode(ode_simple)
    solver.set_integrator('dopri5', max_step=10)

    solver.set_f_params(beta_)

    if t_vals is None:
        t_out = np.linspace(PARAMS.T_1, PARAMS.T_end, 2)
    else:
        t_out = t_vals

    y_out = np.zeros((y0.shape[0], len(t_out)))

    solver.set_initial_value(y0, t_out[0])

    for ind, tt in enumerate(t_out[1:]):
        if solver.successful():
            y_out[:, ind] = solver.y
            solver.integrate(tt)
        else:
            raise RuntimeError('ode solver unsuccessful')

    y_out[:, -1] = solver.y

    return y_out


def find_sev_given_beta_and_no_control(beta_, I0=DEFAULT_I0):

    y_out = find_soln_given_beta_and_no_control(beta_, I0)

    final_S = y_out[0, -1]
    final_I = y_out[1, -1]

    sev = final_I / (final_S + final_I)

    return sev

#
#


def find_beta(final_sev, I0=DEFAULT_I0):

    if final_sev > 1:
        print(f'Warning: {final_sev=}>1')

    INITIAL_GUESS = 0.0078
    # find_beta_vectorised([2e-3, 9.9e-1], I0_value) = [0.00262071, 0.01281348]
    bds = (1e-4, 5e-2)

    min_out = minimize(
        beta_objective,
        [INITIAL_GUESS],
        args=(final_sev, I0),
        bounds=[bds],
        tol=1e-4,
        method='Powell',
    )

    beta_val = min_out.x[0]

    if min_out.success and beta_val < bds[1] and beta_val > bds[0]:
        # print('ok', min_out.x[0])
        return min_out.x[0]

    else:
        print(f'warning!, {min_out.x[0]=}, {final_sev=}')
        return np.nan


def beta_objective(beta_, final_sev, I0):
    F_calculated = find_sev_given_beta_and_no_control(beta_, I0)
    score = (F_calculated - final_sev)**2
    return score

#
#


def host_growth_function(t, S, y):

    if t < PARAMS.T_3:
        senescence = 0
    else:
        # from Elderfield paper
        senescence = (
            0.005 * ((t - PARAMS.T_3)/(2900 - PARAMS.T_3)) +
            + 0.1*exp(-0.02*(2900 - t))
        )

    # r (1-A) from Elderfield paper
    growth = PARAMS.host_growth_rate * (1 - sum(y))

    out = growth - senescence*S

    return out


def yield_function(sev):
    # Load Gam
    filename = 'gam.pickle'
    with open(filename, 'rb') as f:
        gam = pickle.load(f)

    out = gam.predict(sev)[0]

    return out


def economic_yield_function(yield_vec, n_sprays):
    spray_cost = PARAMS.spray_costs[str(n_sprays)]
    profit = [PARAMS.wheat_price*yy - spray_cost for yy in yield_vec]
    return profit


def keys_from_config(config_in):

    sprays = list(map(str, config_in.sprays))
    host = ['Y' if hh else 'N' for hh in config_in.host_on]

    keys = []

    for spray, host in itertools.product(sprays, host):

        spray_str = 'N' if spray == '0' else f'Y{spray}'
        keys.append(f'spray_{spray_str}_host_{host}')

    return keys
#
#

#
#
# Replacement for TraitVecs


def edge_values(n):
    return np.linspace(0, 1, n+1)


def trait_vec(n):
    edge_vals = edge_values(n)

    dx = 0.5*(edge_vals[1] - edge_vals[0])

    vec = np.linspace(dx, 1 - dx, n)

    return vec


def initial_host_dist(n, a, b):
    return beta_dist(n, a, b)


def initial_fung_dist(n, a, b):
    # return beta_dist(n, a, b)
    return gamma_dist(n, a, b)


def gamma_dist(n, a, b):
    """Gamma distribution for fungicide curvatures

    NO LONGER USED?

    See
    - en.wikipedia.org/wiki/Gamma_distribution
    - docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

    Parameters
    ----------
    n : _type_
        _description_
    a : float
        alpha, from shape/rate parameterisation
    b : float
        beta, from shape/rate parameterisation on wiki
    """
    edge_vals = edge_values(n)

    curvature_edge_vals = np.concatenate([[np.inf], np.log(1/edge_vals[1:])])

    dist = [
        gamma.cdf(curvature_edge_vals[i], a, scale=1/b)
        -
        gamma.cdf(curvature_edge_vals[i+1], a, scale=1/b)
        for i in range(len(curvature_edge_vals)-1)
    ]

    out = normalise(dist)

    return out


def beta_dist(n, a, b):
    edge_vals = edge_values(n)

    dist = [
        beta.cdf(edge_vals[i+1], a, b)
        - beta.cdf(edge_vals[i], a, b)
        for i in range(len(edge_vals)-1)
    ]

    out = normalise(dist)

    return out


def initial_point_distribution(n, mean):
    edge_vals = edge_values(n)

    N = len(edge_vals)-1

    j = int(mean * (N-1))

    dist = signal.unit_impulse(N, j)

    out = normalise(dist)

    return out


class Fungicide:

    def __init__(self, num_sprays, dose):

        # different for diff fungicides
        self.decay_rate = FUNG_DECAY_RATE

        self.dose = dose

        if num_sprays == 1:
            self.sprays_list = [PARAMS.T_2]
        elif num_sprays == 2:
            self.sprays_list = [PARAMS.T_2, PARAMS.T_3]
        elif num_sprays == 3:
            self.sprays_list = [PARAMS.T_1, PARAMS.T_2, PARAMS.T_3]
        else:
            self.sprays_list = []

    def effect(self, value_this_strain, t):
        """_summary_

        Parameters
        ----------
        value_this_strain : float
            Trait value between 0 and 1
        t : float
            time (between T_1=1456 and T_end=2515)

        Returns
        -------
        rel_inf_rate
            factor by which infection rate is reduced
        """
        # curvature_this_strain is for a dose of 1

        curvature_this_strain = log(1/value_this_strain)

        concentration = 0

        for T_spray in self.sprays_list:
            if t > T_spray:
                concentration += self.dose * exp(- self.decay_rate*(t-T_spray))

        if concentration == 0:
            return 1

        else:
            rel_inf_rate = exp(- curvature_this_strain*concentration)

            # if have asymptote w
            # out = 1 - w + w*exp(- strain*concentration )
            return rel_inf_rate


def truncated_exp_pdf(x, lambd):
    if x > 100 or x < 0:
        return 0
    else:
        return lambd * np.exp(-lambd*x) / (1 - np.exp(-100*lambd))


def get_dist_mean(dist, traitvec):
    return np.asarray([np.dot(dist[:, yr], traitvec) for yr in range(dist.shape[1])])


def get_host_dist_params_from_config(config):
    """Find a, b from mean, b from config

    Beta distribution in effect space [0,1]

    See en.wikipedia.org/wiki/Beta_distribution

    Parameters
    ----------
    config : Config
        see Config docs

    """

    mu_val = config.l_mu
    b_val = config.l_b

    a_out = (b_val*mu_val)/(1-mu_val)

    return a_out, b_val


def get_fung_dist_params_from_config(config):
    """Find a, b from mean, b from config

    Gamma distribution in curvature space

    NB shape/rate parameterisation from wikipedia:
    - en.wikipedia.org/wiki/Gamma_distribution
    - docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

    This means mu = alpha/beta
    beta = 1/scale

    Parameters
    ----------
    config : Config
        see Config docs

    """

    mu_val = config.k_mu
    b_val = config.k_b

    # GAMMA
    a_out = mu_val*b_val

    return a_out, b_val
