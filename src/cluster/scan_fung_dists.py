"""Fung scan over initial distributions"""

import sys

import numpy as np
import pandas as pd

from poly2.config import Config
from poly2.simulator import SimulatorOneTrait
from poly2.utils import gamma_dist


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
                    a=a,
                    b=b,
                    mu=mu,
                )
            )

            out = pd.concat(
                [
                    out,
                    tmp
                ],
                ignore_index=True
            )

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cf.n_k}_{cf.n_l}'
    out.to_csv(f'../outputs/scan_fung_{conf_str}.csv', index=False)

    return None


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    main(run_index, n_years=100, n_its=5)
