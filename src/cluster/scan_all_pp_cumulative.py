"""For Fung distribution scan, cumulative yield"""

import pandas as pd

from poly2.utils import summarise_by_run_and_year_cumulative

N_RUNS_PER_IT = 100
N_K = 300
N_YEARS = 35
# N_ITS = 100
N_ITS = 5


def combine():

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/scan_all_{ii}_{N_K}_{N_YEARS}_{N_RUNS_PER_IT}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    by_run_year = summarise_by_run_and_year_cumulative(combined)

    print(by_run_year.head())
    print(by_run_year.shape)

    run_info = (
        combined
        .drop(['year', 'yld', 'dose'], axis=1)
        .filter(regex='^(?!(in_)).*$')
        .groupby('run')
        .first()
    )

    print(run_info.head())
    print(run_info.shape)

    out = (
        by_run_year
        .set_index('run')
        .join(run_info)
        .reset_index()
    )

    print(out.head())
    print(out.shape)

    fn = '../outputs/combined/scan_all_cumyld.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
