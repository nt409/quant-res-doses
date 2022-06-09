"""
Fung scan over 'everything' - fung dist, asymptote, decay rate, mutation scale
and proportion.

"""

import sys

import numpy as np
import pandas as pd
from poly2.utils import gamma_dist
from scipy.stats import loguniform

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
        n_l=50,
        verbose=False
    )

    np.random.seed(run)

    out = pd.DataFrame()

    for ii in range(n_its):

        b = np.random.uniform(0, 5)
        mu = np.random.uniform(0, 25)
        a = mu*b

        init_dist = gamma_dist(cf.n_k, a, b)
        #
        #

        asymptote = np.random.uniform(0, 1)
        d_rate_multiplier = np.random.uniform(1/3, 3)

        cf.asymptote = asymptote
        cf.decay_rate = FUNG_DECAY_RATE * d_rate_multiplier
        #
        #

        m_prop_multiplier = loguniform.rvs(1e-1, 10, size=1)
        m_scale_multiplier = loguniform.rvs(1e-1, 10, size=1)

        cf.mutation_proportion = MUTATION_PROP * m_prop_multiplier
        cf.mutation_scale_fung = MUTATION_SCALE * m_scale_multiplier

        for dose in np.linspace(0.1, 1, 10):

            cf.doses = dose*np.ones(cf.n_years)

            sim = SimulatorOneTrait(cf)

            sim.initial_k_dist = init_dist

            data = sim.run_model()

            tmp = (
                pd.DataFrame(
                    dict(
                        yld=data['yield_vec'],
                        year=np.arange(1, 1+len(data['yield_vec']))
                    ))
                .assign(
                    dose=dose,
                    run=run*n_its + ii,
                    mu=mu,
                    b=b,
                    asymptote=asymptote,
                    dec_rate_multiplier=d_rate_multiplier,
                    m_prop_multiplier=m_prop_multiplier,
                    m_scale_multiplier=m_scale_multiplier,
                )
            )

            out = pd.concat([out, tmp], ignore_index=True)

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cf.n_k}_{cf.n_l}'
    out.to_csv(f'../outputs/scan_all_{conf_str}.csv', index=False)

    return None


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    main(run_index, n_years=100, n_its=5)
