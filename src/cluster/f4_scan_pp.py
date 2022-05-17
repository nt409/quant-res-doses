"""For Figure 3"""

import pandas as pd

N_ITS = 200
N_K = 300
N_L = 300


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/fig4_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    grouped = (
        combined
        .groupby(['sprays', 'dose', 'year'])
        .median()
        .reset_index()
        .drop('run', axis=1)
    )

    print(grouped.shape)

    fn = '../outputs/combined/fig4.csv'
    print(f'saving to {fn}')
    grouped.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
