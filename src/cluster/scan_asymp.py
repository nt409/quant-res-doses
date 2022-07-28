"""
Fung scan over 'everything' - fung dist, asymptote, decay rate, mutation scale
and proportion.
"""

from math import exp
import sys

import numpy as np
import pandas as pd
from scipy.stats import loguniform

from poly2.utils import (
    beta_dist,
    edge_values,
    gamma_dist,
    get_dist_mean,
    get_dist_var,
    trait_vec
)
from poly2.config import get_asymptote_config
from poly2.consts import FUNG_DECAY_RATE, MUTATION_PROP, MUTATION_SCALE
from poly2.simulator import SimulatorAsymptote


def main(
    run,
    n_years,
    n_its
):

    cf = get_asymptote_config(
        verbose=False
        n_k=300,
        n_years=n_years,
        k_mu=None,
        k_b=None,
        curvature=None,
    )

    np.random.seed(run)

    out = pd.DataFrame()

    for ii in range(n_its):

        (
            cf,
            mu,
            b,
            curv,
            d_rate_multiplier,
            m_prop_multiplier,
            m_scale_multiplier,
            ME_mean,
            ME_var,
            asymp_dist_summary_dict,
        ) = get_run_params_asymp(cf)

        # * now run the simulation for N doses

        for dose in np.linspace(0.1, 1, 10):

            cf.doses = dose*np.ones(cf.n_years)

            sim = SimulatorAsymptote(cf)

            data = sim.run_model()

            df1 = pd.DataFrame(
                dict(
                    yld=data['yield_vec'],
                    year=np.arange(1, 1+len(data['yield_vec']))
                )
            )

            df2 = pd.DataFrame(
                asymp_dist_summary_dict,
                index=np.arange(df1.shape[0])
            )

            tmp = (
                df1
                .join(df2)
                .assign(
                    dose=dose,
                    run=run*n_its + ii,
                    mu=mu,
                    b=b,
                    curv=curv,
                    dec_rate_multiplier=d_rate_multiplier,
                    m_prop_multiplier=m_prop_multiplier,
                    m_scale_multiplier=m_scale_multiplier,
                    ME_var=ME_var,
                    ME_mean=ME_mean,
                )
            )

            out = pd.concat([out, tmp], ignore_index=True)

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cf.n_k}_{n_years}_{n_its}'
    out.to_csv(f'../outputs/scan_asymp_{conf_str}.csv', index=False)

    return None


def get_run_params_asymp(cf):

    mu = np.random.uniform(1e-2, 1)
    b = np.random.uniform(1e-2, 30)

    a = (b*mu)/(1-mu)

    init_dist = beta_dist(cf.n_k, a, b)

    #
    # * sample fung params

    d_rate_multiplier = np.random.uniform(1/3, 3)
    curv = np.random.uniform(1e-2, 30)

    cf.k_mu = mu
    cf.k_b = b

    cf.curvature = curv
    cf.decay_rate = FUNG_DECAY_RATE * d_rate_multiplier

    #
    # * sample mutation params

    m_prop_multiplier = float(loguniform.rvs(1e-1, 10, size=1))
    m_scale_multiplier = float(loguniform.rvs(1e-1, 10, size=1))

    cf.mutation_proportion = MUTATION_PROP * m_prop_multiplier
    cf.mutation_scale_fung = MUTATION_SCALE * m_scale_multiplier

    # * in trait space, get mean and variance

    tv = trait_vec(cf.n_k)
    # NB get_dist_var/mean needs init_dist to be shape (n_k, n_years)
    reshaped = init_dist.reshape((cf.n_k, 1))
    tv_var = get_dist_var(reshaped, tv)[0]
    tv_mean = get_dist_mean(reshaped, tv)[0]

    # max_effect = 1 + w * (exp(-curv) - 1)
    max_effect_mean = 1 + tv_mean * (exp(-curv) - 1)
    max_effect_var = ((exp(-curv) - 1)**2) * tv_var

    # * now get info on density in 0-0.1, 0.1-0.2, ... 0.9-1
    ev = edge_values(10)

    dist_summary = beta_dist(10, a, b)

    dist_summary_keys = [(
        f'in_{ev[ii]:.1f}_{ev[ii+1]:.1f}'.replace('.', 'p')
    )
        for ii in range(len(ev)-1)
    ]

    asymp_dist_summary_dict = {
        dist_summary_keys[ii]: dist_summary[ii] for ii in range(10)
    }

    return (
        cf,
        mu,
        b,
        curv,
        d_rate_multiplier,
        m_prop_multiplier,
        m_scale_multiplier,
        max_effect_mean,
        max_effect_var,
        asymp_dist_summary_dict,
    )


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    # proper run:
    main(run_index, n_years=35, n_its=100)

    # test run:
    # main(run_index, n_years=10, n_its=1)
