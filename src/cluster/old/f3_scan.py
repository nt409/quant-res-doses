"""For Figure 3"""

from math import floor
import sys

import numpy as np
import pandas as pd

from poly2.config import Config
from poly2.run import no_joblib_single_run


def main(
    run,
    n_years=15
):

    cmh = Config(
        type='single',
        sprays=[1, 2, 3],
        host_on=[True],
        n_years=n_years,
        n_k=300,
        n_l=300,
    )

    # don't want means to go to 1
    host_means = np.linspace(0.01, 0.99, 20)
    doses = np.linspace(0, 1, 20)

    host_ind = floor(run/len(doses))
    dose_ind = run % len(doses)

    cmh.l_mu = host_means[host_ind]
    dose = doses[dose_ind]

    data_out = no_joblib_single_run(
        cmh,
        doses=dose*np.ones(n_years)
    )

    output1 = data_out['spray_1_host_Y']
    output2 = data_out['spray_2_host_Y']
    output3 = data_out['spray_3_host_Y']

    out = (
        pd.concat(
            [
                get_dataframe(output1, 1, dose, host_means[host_ind]),
                get_dataframe(output2, 2, dose, host_means[host_ind]),
                get_dataframe(output3, 3, dose, host_means[host_ind])
            ],
            ignore_index=True
        )
        .reset_index(drop=True)
    )

    conf_str = f'{run}_{cmh.n_k}_{cmh.n_l}'

    out.to_csv(f'../outputs/fig3_{conf_str}.csv', index=False)

    return None


def get_dataframe(output, sprays, dose, host_mean):
    """Get dataframe summarising output

    Parameters
    ----------
    output : list of dicts
        --
    sprays : int
        --
    dose : float
        --
    host_mean : float
        --

    Returns
    -------
    out : pd.DataFrame
        Columns:
        - host_mu
        - sprays
        - dose
        - year
        - sev
        - yld
        - econ

    """

    out = pd.DataFrame(
        {
            'host_mu': host_mean,
            'sprays': int(sprays),
            'dose': dose,
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
