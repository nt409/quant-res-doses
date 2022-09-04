import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from poly2.config import Config
from poly2.simulator import SimulatorOneTrait


def main(ii):

    cf = Config(
        verbose=False,
        n_k=100,
        n_years=20,
    )

    cf.mutation_proportion = 0
    cf.mutation_scale_fung = 1
    cf.mutation_scale_host = 1

    pr = 1e-3

    out = pd.DataFrame()

    sim = SimulatorOneTrait(cf)
    k_vec = sim.k_vec

    indices = np.arange(0, 100, 1)

    for jj in tqdm(indices):

        if k_vec[jj] <= k_vec[ii]:
            continue

        init_dist_1 = np.zeros(cf.n_k)
        init_dist_1[ii] = 1-pr
        init_dist_1[jj] = pr

        sim = SimulatorOneTrait(cf)

        tmp = (
            get_mono_data(cf, init_dist_1)
            .assign(
                s=sim.k_vec[ii],
                r=sim.k_vec[jj],
            )
            .drop(['mean'], axis=1)
        )

        out = pd.concat([out, tmp])

    out = out.reset_index(drop=True)

    out.to_csv(f'../outputs/f2/run_{ii}.csv', index=False)

    return None


def get_mono_data(cf, init_dist):

    doses = np.arange(0.1, 1.1, 0.1)

    df = pd.DataFrame()

    for dd in doses:
        cf.doses = dd*np.ones(cf.n_years)

        sim = SimulatorOneTrait(cf)

        sim.initial_k_dist = init_dist

        data = sim.run_model()

        tmp = pd.DataFrame(
            dict(
                yld=data['yield_vec'],
                dose=dd,
                year=data['year'],
                mean=data['fung_mean'][:-1],
            ))

        df = pd.concat([df, tmp])

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    main(run_index)
