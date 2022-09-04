"""Combine dataframes from cluster run"""

import pandas as pd

N_ITS = 99


def combine():

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/f2/run_{ii}.csv')
        for ii in range(N_ITS)
    ]).reset_index(drop=True)

    print(combined.shape)

    out = (
        combined
        .groupby(['year', 's', 'r'])
        .apply(lambda x: x.loc[x.yld.idxmax()])
        .drop(['s', 'r', 'year'], axis=1)
        .reset_index()
    )

    fn = '../outputs/combined/fig2.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
