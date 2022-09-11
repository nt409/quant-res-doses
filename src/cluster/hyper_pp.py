"""Combine hyperoptimisations"""

import pandas as pd

N_ITS = 50


def combine(model, folder):

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/{folder}/{model}_{ii}.csv')
        for ii in range(N_ITS)
    ]).reset_index(drop=True)

    print(combined.shape)

    fn = f'../outputs/combined/{folder}/{model}.csv'
    print(f'saving to {fn}')
    combined.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    MODEL = 'all'
    combine(MODEL, 'scores')
    combine(MODEL, 'hyperparams')
