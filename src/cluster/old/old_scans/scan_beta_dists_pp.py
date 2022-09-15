"""For Fung distribution scan"""

import pandas as pd

from poly2.utils import summarise_by_run_and_year

N_ITS = 200
N_K = 300
N_L = 50


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_dist_beta_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    by_run_year = summarise_by_run_and_year(combined)

    run_info = (
        combined
        .groupby('run').mean()
        .loc[:, ['mu', 'b']]
    )

    out = (
        by_run_year
        .set_index('run')
        .join(run_info)
        .reset_index()
    )

    print(out.shape)

    fn = '../outputs/combined/beta_scan.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
