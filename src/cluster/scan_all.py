"""
Fung scan over 'everything' - fung dist, asymptote, decay rate, mutation scale
and proportion.

"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import loguniform

from poly2.utils import (
    edge_values,
    gamma_dist,
    get_dist_mean,
    get_dist_var,
    trait_vec
)
from poly2.config import Config
from poly2.consts import FUNG_DECAY_RATE, MUTATION_PROP, MUTATION_SCALE
from poly2.simulator import SimulatorOneTrait


def main(
    run,
    n_years,
    n_its
):

    cf = Config(
        n_years=n_years,
        n_k=300,
        verbose=False
    )

    np.random.seed(run)

    out = pd.DataFrame()

    for ii in range(n_its):

        b = np.random.uniform(0, 5)
        mu = np.random.uniform(0, 25)
        a = mu*b
        init_dist = gamma_dist(cf.n_k, a, b)

        gamma_var = a/b**2

        #
        # * sample fung params

        asymptote = np.random.uniform(0, 1)
        d_rate_multiplier = np.random.uniform(1/3, 3)

        cf.asymptote = asymptote
        cf.decay_rate = FUNG_DECAY_RATE * d_rate_multiplier

        #
        # * sample mutation params

        m_prop_multiplier = float(loguniform.rvs(1e-1, 10, size=1))
        m_scale_multiplier = float(loguniform.rvs(1e-1, 10, size=1))

        cf.mutation_proportion = MUTATION_PROP * m_prop_multiplier
        cf.mutation_scale_fung = MUTATION_SCALE * m_scale_multiplier

        # * in trait space, get mean and variance
        tv = trait_vec(cf.n_k)
        reshaped = init_dist.reshape((cf.n_k, 1))
        tv_var = get_dist_var(reshaped, tv)[0]
        tv_mean = get_dist_mean(reshaped, tv)[0]

        # * now get info on density in 0-0.1, 0.1-0.2, ... 0.9-1
        ev = edge_values(10)

        dist_summary = gamma_dist(10, a, b)

        dist_summary_keys = [(
            f'in_{ev[ii]:.1f}_{ev[ii+1]:.1f}'.replace('.', 'p')
        )
            for ii in range(len(ev)-1)
        ]

        dist_summary_dict = {
            dist_summary_keys[ii]: dist_summary[ii] for ii in range(10)
        }

        #
        #

        # * now run the simulation for N doses

        for dose in np.linspace(0.1, 1, 10):

            cf.doses = dose*np.ones(cf.n_years)

            sim = SimulatorOneTrait(cf)

            sim.initial_k_dist = init_dist

            data = sim.run_model()

            df1 = pd.DataFrame(
                dict(
                    yld=data['yield_vec'],
                    year=np.arange(1, 1+len(data['yield_vec']))
                )
            )

            df2 = pd.DataFrame(
                dist_summary_dict,
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
                    asymptote=asymptote,
                    dec_rate_multiplier=d_rate_multiplier,
                    m_prop_multiplier=m_prop_multiplier,
                    m_scale_multiplier=m_scale_multiplier,
                    gamma_var=gamma_var,
                    tv_var=tv_var,
                    tv_mean=tv_mean,
                )
            )

            out = pd.concat([out, tmp], ignore_index=True)

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cf.n_k}_{n_years}_{n_its}'
    out.to_csv(f'../outputs/scan_all_{conf_str}.csv', index=False)

    return None


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    # main(run_index, n_years=100, n_its=5)
    main(run_index, n_years=20, n_its=25)
    # main(run_index, n_years=5, n_its=2)
