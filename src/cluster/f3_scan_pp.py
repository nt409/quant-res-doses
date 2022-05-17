"""For Figure 3"""

import pandas as pd

N_ITS = 190
N_K = 300
N_L = 300


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/fig3_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    # grouped = (
    #     combined
    #     .groupby(['sprays', 'dose', 'year'])
    #     .median()
    #     .reset_index()
    # )

    # print(grouped.shape)

    fn = '../outputs/combined/fig3.csv'
    print(f'saving to {fn}')
    combined.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
