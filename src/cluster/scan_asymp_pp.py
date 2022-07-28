"""For Fung distribution scan"""

import pandas as pd

from poly2.utils import summarise_by_run_and_year

N_RUNS_PER_IT = 100
N_K = 300
N_YEARS = 35
N_ITS = 100


def combine():

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/scan_asymp_{ii}_{N_K}_{N_YEARS}_{N_RUNS_PER_IT}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    by_run_year = summarise_by_run_and_year(combined)

    run_info = (
        combined
        .groupby('run').mean()
        .drop(['year', 'yld', 'dose'], axis=1)
    )

    print(run_info.columns)
    print(by_run_year.columns)

    out = (
        by_run_year
        .set_index('run')
        .join(run_info)
        .reset_index()
    )

    print(out.shape)

    fn = '../outputs/combined/scan_asymp.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
