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
import pandas as pd

from scipy.stats import beta, gamma, norm
from scipy.optimize import minimize
from scipy.integrate import ode
from scipy import signal

from poly2.consts import DEFAULT_I0, FUNG_DECAY_RATE
from poly2.params import PARAMS


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


def yield_fn(sev):
    # Load GAM
    filename = 'gam.pickle'
    with open(filename, 'rb') as f:
        gam = pickle.load(f)

    out = gam.predict(sev)[0]

    return out


def economic_yield(yield_vec, sprays_vec, doses):

    yield_vec = np.array(yield_vec)
    sprays_vec = np.array(sprays_vec)
    doses = np.array(doses)

    did_apply_bool = doses > 0

    applied = did_apply_bool.astype(int)

    cost_application = PARAMS.application_cost_per_spray * sprays_vec * applied
    cost_fungicide = PARAMS.chemical_cost_per_spray * sprays_vec * doses

    revenue = PARAMS.wheat_price * yield_vec

    profit = revenue - cost_application - cost_fungicide

    return profit


def economic_yield_mixture(yield_vec, sprays_vec, doses_A, doses_B):

    yield_vec = np.array(yield_vec)
    sprays_vec = np.array(sprays_vec)
    doses_A = np.array(doses_A)
    doses_B = np.array(doses_B)

    did_apply_bool_A = doses_A > 0
    did_apply_bool_B = doses_B > 0

    applied_A = did_apply_bool_A.astype(int)
    applied_B = did_apply_bool_B.astype(int)

    applied = (applied_A + applied_B)
    applied[applied > 0] = 1

    cost_application = PARAMS.application_cost_per_spray * sprays_vec * applied

    cost_fungicide = PARAMS.chemical_cost_per_spray * sprays_vec * (
        doses_A + doses_B
    )

    revenue = PARAMS.wheat_price * yield_vec

    profit = revenue - cost_application - cost_fungicide

    return profit


def disease_severity(final_Is, final_Ss):

    final_Is = np.array(final_Is)
    final_Ss = np.array(final_Ss)

    # scale so proportion of final leaf size
    sev = final_Is / (final_Is + final_Ss)

    return sev


def keys_from_config(config_in):

    sprays = list(map(str, config_in.sprays))
    host = ['Y' if hh else 'N' for hh in config_in.host_on]

    keys = []

    for spray, host in itertools.product(sprays, host):

        spray_str = 'N' if spray == '0' else spray
        keys.append(f'spray_{spray_str}_host_{host}')

    return keys

#
#
# Distribution stuff:


def edge_values(n):
    return np.linspace(0, 1, n+1)


def trait_vec(n):
    edge_vals = edge_values(n)

    dx = 0.5*(edge_vals[1] - edge_vals[0])

    vec = np.linspace(dx, 1 - dx, n)

    return vec


# def initial_host_dist(n, a, b):
#     return beta_dist(n, a, b)


# def initial_fung_dist(n, a, b):
#     return gamma_dist(n, a, b)


def gamma_dist(n, a, b):
    """Gamma distribution for fungicide curvatures

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


#
#
#

# Fungicide stuff

class Fungicide:

    def __init__(self, num_sprays, dose, decay_rate=None, asymptote=None):
        """init method

        Fungicide for a single year

        Needs decay_rate=None not =FUNG_DECAY_RATE, since input often 'None'
        rather than not included

        Parameters
        ----------
        num_sprays : int
            number of sprays per year
        dose : float
            dose applied
        decay_rate : float, optional
            fungicide decay rate, default FUNG_DECAY_RATE if input was None
        asymptote : float, optional
            in (0,1), default is 1 if input was None
        """

        if decay_rate is None:
            self.decay_rate = FUNG_DECAY_RATE
        else:
            self.decay_rate = decay_rate

        if asymptote is None:
            self.asymptote = 1
        else:
            self.asymptote = asymptote

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
        """Effect of fungicide at time t on a particular strain

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

        curv = log(1/value_this_strain)

        # if have asymptote w
        w = self.asymptote

        concentration = 0

        for T_spray in self.sprays_list:
            if t > T_spray:
                concentration += self.dose * exp(-self.decay_rate*(t-T_spray))

        if concentration == 0:
            return 1

        else:
            # rel_inf_rate = exp(- curv*concentration)

            rel_inf_rate = 1 - w + w*exp(- curv*concentration)

            return rel_inf_rate


class FungicideAsymptote:
    """Parameterised with type 1 partial resistance, i.e. curvature fixed and
    asymptote depends on the value of k
    """

    def __init__(self, num_sprays, dose, curvature, decay_rate=None):
        """init method

        Fungicide for a single year

        Needs decay_rate=None not =FUNG_DECAY_RATE, since input often 'None'
        rather than not included

        Parameters
        ----------
        num_sprays : int
            number of sprays per year
        dose : float
            dose applied
        curvature : float
            curvature param (const)
        decay_rate : float, optional
            fungicide decay rate, default FUNG_DECAY_RATE if input was None
        """

        if decay_rate is None:
            self.decay_rate = FUNG_DECAY_RATE
        else:
            self.decay_rate = decay_rate

        self.dose = dose

        self.curv = curvature

        if num_sprays == 1:
            self.sprays_list = [PARAMS.T_2]
        elif num_sprays == 2:
            self.sprays_list = [PARAMS.T_2, PARAMS.T_3]
        elif num_sprays == 3:
            self.sprays_list = [PARAMS.T_1, PARAMS.T_2, PARAMS.T_3]
        else:
            self.sprays_list = []

    def effect(self, value_this_strain, t):
        """Effect of fungicide at time t on a particular strain

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

        # asymptote w
        w = value_this_strain

        concentration = 0

        for T_spray in self.sprays_list:
            if t > T_spray:
                concentration += self.dose * exp(-self.decay_rate*(t-T_spray))

        if concentration == 0:
            return 1

        else:
            # rel_inf_rate = exp(- curv*concentration)

            rel_inf_rate = 1 - w + w*exp(- self.curv*concentration)

            return rel_inf_rate


class FungicideNoDecay:

    def __init__(self, num_sprays, dose):
        """init method

        Fungicide for a single year

        Parameters
        ----------
        num_sprays : int
            number of sprays per year
        dose : float
            dose applied
        """

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
        """Effect of fungicide at time t on a particular strain

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

        curvature_this_strain = log(1/value_this_strain)

        concentration = 0

        for T_spray in self.sprays_list:
            if t > T_spray and t < T_spray+240:
                concentration = self.dose

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
    """Get vector of fung or host distribution mean each year

    Parameters
    ----------
    dist : np.array
        e.g. output['spray_1_host_N']['fung_dists']
    traitvec : np.array
        output['spray_1_host_N']['k_vec']

    Returns
    -------
    out : np.array
        length = number of years in dist, which includes year 0 and N
    """
    means = np.asarray([np.dot(dist[:, yr], traitvec)
                       for yr in range(dist.shape[1])])
    return means


def get_dist_var(dist, traitvec):
    """Dist variance

    Parameters
    ----------
    dist : np.array
        e.g. output['spray_1_host_N']['fung_dists']
    traitvec : np.array
        output['spray_1_host_N']['k_vec']

    Returns
    -------
    out : np.array
        length = number of years in dist, which includes year 0 and N
    """
    means = get_dist_mean(dist, traitvec)

    trait_n = dist.shape[0]
    n_years = dist.shape[1]

    variances = np.zeros(n_years)
    for yy in range(n_years):
        for dd in range(trait_n):
            variances[yy] += dist[dd, yy]*(traitvec[dd] - means[yy])**2

    return variances

# For simulator:


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


def get_dispersal_kernel(vec, p, mutation_scale):

    N = len(vec)

    kernel = np.zeros((N, N))

    for parent in range(N):
        # some proportion stays at position i
        not_dispersing = signal.unit_impulse(N, parent)

        dispersing = dispersal(vec, parent, mutation_scale)

        kernel[:, parent] = p*dispersing + (1-p)*not_dispersing

    return kernel


def dispersal(vec, parent_index, mut_scale):

    stan_dev = mut_scale**0.5

    edges = edge_values(len(vec))

    disp = norm.cdf(edges, loc=vec[parent_index], scale=stan_dev)

    dispersing = np.diff(disp)

    top = 1 - disp[-1]

    bottom = disp[0]

    dispersing[0] += bottom

    dispersing[-1] += top

    return dispersing


def get_model_times():
    times = np.linspace(PARAMS.T_1, PARAMS.T_end, 100)
    times = list(times)

    times += [PARAMS.T_2, PARAMS.T_3]
    times = sorted(times)
    times = np.asarray(times)
    return times


#
#
# Post process cluster stuff
#
def monotonic_yld(df):
    du = df.sort_values('dose')
    diffs = du.yld.diff()
    return sum(diffs > 0)


def best_dose(df):
    du = df.sort_values('yld', ascending=False)
    out = float(du.dose.iloc[0])
    return out


def summarise_by_run_and_year(combined):
    """Return dataframe with best dose and number of doses for which 
    yld[i+1]>yld[i]

    Parameters
    ----------
    combined : pd.DataFrame
        From one of the parameter scans, columns:
        - run
        - year
        - dose
        - yld

    Returns
    -------
    by_run_year : pd.DataFrame
        columns:
        - run
        - year
        - best_dose
        - n_pos_diff (<=N_doses-1 if monotonic inc in dose, less if not)
    """
    yld_diffs = (
        combined
        .groupby(['run', 'year'])
        .apply(monotonic_yld)
        .reset_index()
        .rename(columns={0: 'n_pos_diff'})
    )

    best_doses = (
        combined
        .groupby(['run', 'year'])
        .apply(best_dose)
        .reset_index()
        .rename(columns={0: 'best_dose'})
    )

    by_run_year = (
        best_doses.set_index(['run', 'year'])
        .join(
            yld_diffs.set_index(['run', 'year'])
        )
        .reset_index()
    )

    return by_run_year


# for fitting
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
