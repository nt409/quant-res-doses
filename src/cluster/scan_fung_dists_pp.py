"""For Fung distribution scan"""

import pandas as pd

N_ITS = 200
N_K = 30
N_L = 50


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_fung_{ii}_{N_K}_{N_L}.csv')
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

    diffs.set_index('run').join(
        combined.groupby('run').mean().reset_index().loc[:, ['mu', 'b']]
    )

    print(diffs.shape)

    fn = '../outputs/combined/fung_scan.csv'
    print(f'saving to {fn}')
    diffs.to_csv(fn, index=False)

    return None


def monotonic_yld(df):
    du = df.sort_values('dose')
    diffs = du.yld.diff()
    return sum(diffs > 0)


if __name__ == "__main__":
    combine()
