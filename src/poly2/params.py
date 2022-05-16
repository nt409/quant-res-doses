import pandas as pd


class Constants:
    def __init__(self):
        """
        Running params, setup etc
        """

        consts = pd.read_csv('../data/constants.csv')

        self.wheat_price = float(
            consts[consts['Parameter'] == 'Price of wheat'].Value
        )

        self.chemical_cost_per_spray = float(
            consts[consts['Parameter'] == 'cost_per_spray_fungicide'].Value)
        self.application_cost_per_spray = float(
            consts[consts['Parameter'] == 'cost_per_spray_application'].Value)

        # times
        self.T_1 = float(consts[consts['Parameter'] == 'T1'].Value)
        self.T_2 = float(consts[consts['Parameter'] == 'T2'].Value)
        self.T_3 = float(consts[consts['Parameter'] == 'T3'].Value)
        self.T_end = float(consts[consts['Parameter'] == 'T_end'].Value)

        # host growth
        self.host_growth_rate = float(
            consts[consts['Parameter'] == 'Host_growth_rate'].Value
        )

        self.host_growth_initial_area = float(
            consts[consts['Parameter'] == 'Host_growth_initial_area'].Value
        )

        # # mapping between strat and index
        self.strategy_mapping = dict(
            spray_N_host_N=0,
            spray_N_host_Y=1,
            spray_1_host_N=2,  # spray_Y1_host_N=2,
            spray_1_host_Y=3,  # spray_Y1_host_Y=3,
            spray_2_host_N=4,  # spray_Y2_host_N=4,
            spray_2_host_Y=5,  # spray_Y2_host_Y=5,
            spray_3_host_N=6,  # spray_Y3_host_N=6,
            spray_3_host_Y=7  # spray_Y3_host_Y=7
        )


PARAMS = Constants()
