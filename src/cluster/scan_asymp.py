"""Fung scan over initial distributions"""

import sys

import numpy as np
import pandas as pd

from poly2.config import get_asymptote_config
from poly2.simulator import SimulatorAsymptote


def main(
    run,
    n_years,
    n_its
):

    cf = get_asymptote_config(
        verbose=False,
        n_k=300,
        n_years=n_years,
        k_mu=None,
        k_b=None,
        curvature=None,
    )

    np.random.seed(run)

    out = pd.DataFrame()

    for ii in range(n_its):

        mu = np.random.uniform(1e-1, 1)
        b = np.random.uniform(1e-2, 30)
        curv = np.random.uniform(1e-2, 30)

        cf.mu = mu
        cf.b = b
        cf.curv = curv

        for dose in np.linspace(0.1, 1, 10):

            cf.doses = dose*np.ones(cf.n_years)

            sim = SimulatorAsymptote(cf)

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
                    b=b,
                    mu=mu,
                    curv=curv,
                )
            )

            out = pd.concat([out, tmp], ignore_index=True)

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cf.n_k}_{n_years}'
    out.to_csv(f'../outputs/scan_asymp_{conf_str}.csv', index=False)

    return None


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    # main(run_index, n_years=1, n_its=1)
    main(run_index, n_years=100, n_its=5)
