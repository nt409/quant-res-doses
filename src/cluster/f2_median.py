"""For Figure 2"""

import sys

import numpy as np
import pandas as pd

from poly2.config import Config
from poly2.run import no_joblib_multiple_run


def main(
    run,
    n_its=5,
    n_years=15
):

    cmh = Config(
        type='multi',
        sprays=[1, 2, 3],
        host_on=[False],
        n_years=n_years,
        n_k=500,
        n_l=50,
        n_iterations=n_its,
    )

    cmh.beta_multi = cmh.beta_multi[run*n_its*n_years:(1+run)*n_its*n_years]

    out = pd.DataFrame()

    for dose in np.linspace(0.1, 1, 10):

        multi_runs = no_joblib_multiple_run(
            cmh,
            doses=dose*np.ones(n_years)
        )

        output1 = multi_runs['spray_1_host_N']
        output2 = multi_runs['spray_2_host_N']
        output3 = multi_runs['spray_3_host_N']

        out = pd.concat(
            [
                out,
                get_dataframe(output1, run, 1, dose, n_its),
                get_dataframe(output2, run, 2, dose, n_its),
                get_dataframe(output3, run, 3, dose, n_its)
            ],
            ignore_index=True
        )

    out = out.reset_index(drop=True)

    conf_str = f'{run}_{cmh.n_k}_{cmh.n_l}'
    out.to_csv(f'../outputs/fig2_{conf_str}.csv')

    return None


def get_dataframe(output, run, sprays, dose, n_its):
    """Get dataframe summarising output

    Parameters
    ----------
    output : list of dicts
        --
    run : int
        --
    sprays : int
        --
    dose : float
        --
    n_its : int
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

    out = pd.DataFrame()

    for ii in range(n_its):
        index = n_its*run + ii

        tmp = pd.DataFrame(
            {
                'run': index,
                'sprays': int(sprays),
                'dose': dose,
                'year': np.arange(1, 1+len(output[ii]['yield_vec'])),
                'sev': 100*output[ii]['dis_sev'],
                'yld': output[ii]['yield_vec'],
                'econ': output[ii]['econ'],
            }
        )

        out = pd.concat([out, tmp])

    return out


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Supply one argument: a run index")

    run_index = int(sys.argv[1])

    main(run_index, n_its=5, n_years=15)
