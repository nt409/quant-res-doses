"""For Fung distribution scan"""

import pandas as pd

from poly2.utils import best_dose, monotonic_yld

N_ITS = 200
N_K = 300
N_L = 50


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_fung_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

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
        best_doses
        .set_index(['run', 'year'])
        .join(
            yld_diffs
            .set_index(['run', 'year'])
        )
    )

    run_info = (
        combined
        .groupby('run').mean()
        .loc[:, ['mu', 'b']]
    )

    print(by_run_year.head())

    out = (
        by_run_year
        .set_index('run')
        .join(run_info)
    )

    print(out.shape)

    fn = '../outputs/combined/fung_scan.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
