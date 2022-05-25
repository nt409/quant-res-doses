"""For Figure 1"""

import pandas as pd


def combine(n_its, n_k):

    combined = pd.concat([
        pd.read_csv(f'../outputs/fig2_{ii}_{n_k}.csv')
        for ii in range(n_its)
    ])

    print(combined.shape)

    fn = '../outputs/combined/fig2.csv'
    print(f'saving to {fn}')
    combined.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    N_ITS = 440
    N_K = 300

    combine(N_ITS, N_K)
