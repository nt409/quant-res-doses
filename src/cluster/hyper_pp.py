"""Combine hyperoptimisations"""

import sys
import pandas as pd

N_ITS = 200


def combine(model):

    combined = pd.concat([
        pd.read_csv(
            f'../outputs/hyperparams/{model}_{ii}.csv')
        for ii in range(N_ITS)
    ]).reset_index(drop=True)

    print(combined.shape)

    fn = f'../outputs/combined/hyperparams/{model}.csv'
    print(f'saving to {fn}')
    combined.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    # MODEL = 'all' / Y10 / cumulative / asymp

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: the model name")

    MODEL = sys.argv[1]

    combine(MODEL)
