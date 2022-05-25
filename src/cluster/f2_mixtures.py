"""For Figure 2"""

from math import floor
import sys

import numpy as np
import pandas as pd

from poly2.config import ConfigMixture
from poly2.run import no_joblib_simulations_run


def main(run, n_years=15):

    doses = np.linspace(0, 1, 21)

    ind_A = floor(run/21)
    ind_B = run % 21

    da = doses[ind_A]
    db = doses[ind_B]

    cf = ConfigMixture(
        sprays=[2],
        n_years=n_years,
        n_k=300,
        dose_A=da,
        dose_B=db,
    )

    data = no_joblib_simulations_run(cf)

    output = data['spray_2_host_N']

    out = get_dataframe(output, run, da, db)

    conf_str = f'{run}_{cf.n_k}'
    out.to_csv(f'../outputs/fig2_{conf_str}.csv', index=False)

    return None


def get_dataframe(output, run, dose_A, dose_B):
    """Get dataframe summarising output

    Parameters
    ----------
    output : list of dicts
        --
    run : int
        --
    dose_A : float
        --
    dose_A : float
        --

    Returns
    -------
    out : pd.DataFrame
        Columns:
        - run
        - sprays
        - dose
        - year
        - sev
        - yld
        - econ

    """

    out = pd.DataFrame(
        {
            'run': run,
            'dose_A': dose_A,
            'dose_B': dose_B,
            'year': np.arange(1, 1+len(output['yield_vec'])),
            'sev': 100*output['dis_sev'],
            'yld': output['yield_vec'],
            'econ': output['econ'],
        }
    )

    return out


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    main(run_index, n_years=15)
