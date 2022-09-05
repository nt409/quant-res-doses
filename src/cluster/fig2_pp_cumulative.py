"""Combine dataframes from cluster run"""

import numpy as np
import pandas as pd

N_ITS = 99
# final one has no entries
# N_ITS = 5


def combine():

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/f2/run_{ii}.csv')
        for ii in range(N_ITS)
    ]).reset_index(drop=True)

    print(combined.shape)
    print(combined.head(20))

    out = (
        combined
        .groupby(['s', 'r', 'dose'])
        .apply(lambda df: pd.DataFrame(
            dict(
                year=df.year,
                r=df.r,
                s=df.s,
                dose=df.dose,
                cumyld=np.cumsum(df.yld),
            )
        ))
        .groupby(['s', 'r', 'year'])
        .apply(lambda x: x.loc[x.cumyld.idxmax()])
        .reset_index(drop=True)
        .rename(columns={'dose': 'best_dose'})
    )

    print(out.head(20))

    fn = '../outputs/combined/fig2_cumyld.csv'
    print(f'saving to {fn}')
    out.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
