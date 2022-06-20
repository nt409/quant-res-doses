"""For Fung distribution scan"""

import pandas as pd

from poly2.utils import summarise_by_run_and_year

N_ITS = 200
N_K = 300
N_YEARS = 20
N_ITS = 25


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_all_{ii}_{N_K}_{N_YEARS}_{N_ITS}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    by_run_year = summarise_by_run_and_year(combined)

    run_info = (
        combined
        .groupby('run').mean()
        .loc[:, [
            'mu',
            'b',
            'asymptote',
            'dec_rate_multiplier',
            'm_prop_multiplier',
            'm_scale_multiplier',
        ]]
    )

    out = (
        by_run_year
        .set_index('run')
        .join(run_info)
        .reset_index()
    )

    print(out.shape)

    fn = '../outputs/combined/scan_all.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
