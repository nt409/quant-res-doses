"""For Fung distribution scan"""

import pandas as pd

from poly2.utils import monotonic_yld

N_ITS = 200
N_K = 300
N_L = 50


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_mut_pars_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    diffs = (
        combined
        .groupby(['run', 'year'])
        .apply(monotonic_yld)
        .reset_index()
        .rename(columns={0: 'n_pos_diff'})
    )

    out = (
        diffs
        .set_index('run')
        .join(
            combined
            .groupby('run').mean()
            .loc[:, ['m_prop_multiplier', 'm_scale_multiplier']]
        )
    )

    print(out.shape)

    fn = '../outputs/combined/mut_pars_scan.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
