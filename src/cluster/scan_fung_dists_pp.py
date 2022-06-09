"""For Fung distribution scan"""

import pandas as pd

N_ITS = 200
N_K = 30
N_L = 50


def combine():

    combined = pd.concat([
        pd.read_csv(f'../outputs/scan_fung_{ii}_{N_K}_{N_L}.csv')
        for ii in range(N_ITS)
    ])

    print(combined.shape)

    fn = '../outputs/combined/fung_scan.csv'
    print(f'saving to {fn}')
    combined.to_csv(fn, index=False)

    return None


if __name__ == "__main__":
    combine()
