# -*- coding: utf-8 -*-
# creat_time: 2021/10/26 21:42

### Generate macro variables and technical indicators ###
# Modified based on Neely, C. J., Rapach, D. E., Tu, J., & Zhou, G. (2014).

import pandas as pd
import numpy as np

predictor_raw = pd.read_csv("PredictorData1926-2020.csv")
predictor_raw.tail()
predictor_raw.columns


## Generating equity risk premium, 1927:01-2020:12
n_rows = predictor_raw.shape[0]
market_return = predictor_raw['CRSP_SPvw'][1:].values
risk_free_lag = predictor_raw['Rfree'][0:(n_rows - 1)].values
log_equity_premium = np.log(1 + market_return) - np.log(1 + risk_free_lag)
equity_premium = market_return - risk_free_lag


### Generating 12 macroeconomic variables, 1927:01-2020:12
# Notes: we exclude the log dividend-earnings ratio (DE) and the long-term yield (LTY).

# (1) Dividend-price ratio (log), DP
D12 = predictor_raw['D12'][1:].values
SP500 = predictor_raw['Index'][1:].values
DP = np.log(D12) - np.log(SP500)

# (2) Dividend yield (log), DY
SP500_lag = predictor_raw['Index'][0:(n_rows - 1)].values
DY = np.log(D12) - np.log(SP500_lag)

# (3) Earnings-price ratio (log), EP
E12 = predictor_raw['E12'][1:].values
EP = np.log(E12) - np.log(SP500)

# (4) stock variance, SVAR
SVAR = predictor_raw['svar'][1:].values

# (5) Book-to-market ratio, BM
BM = predictor_raw['b/m'][1:].values

# (6) Net equity expansion, NTIS
NTIS = predictor_raw['ntis'][1:].values

# (7) Treasury bill rate (annual %), TBL
TBL = predictor_raw['tbl'][1:].values
TBL = 100 * TBL

# (8) Long-term return (%), LTR
LTR = predictor_raw['ltr'][1:].values
LTR = 100 * LTR

# (9) Term spread (annual %), TMS
LTY = predictor_raw['lty'][1:].values
LTY = 100 * LTY
TMS = LTY - TBL

# (10) Default yield spread, DFY
AAA = predictor_raw['AAA'][1:].values
BAA = predictor_raw['BAA'][1:].values
DFY = 100 * (BAA - AAA)

# (11) Default return spread, DFR
CORPR = predictor_raw['corpr'][1:].values
DFR = 100 * CORPR - LTR

# (12) Inflation (%, lagged), INFL
INFL = predictor_raw['infl'][0:(n_rows - 1)].values
INFL = 100 * INFL

## Collect 12 macroeconomic variables
macro = np.concatenate([DP.reshape(-1, 1), DY.reshape(-1, 1), EP.reshape(-1, 1),
                        SVAR.reshape(-1, 1), BM.reshape(-1, 1), NTIS.reshape(-1, 1),
                        TBL.reshape(-1, 1), LTR.reshape(-1, 1), TMS.reshape(-1, 1),
                        DFY.reshape(-1, 1), DFR.reshape(-1, 1), INFL.reshape(-1, 1)], axis=1)
macro.shape
#


## Collect 12 technical indicators
technical = predictor_raw[['MA_1_9', 'MA_1_12', 'MA_2_9', 'MA_2_12', 'MA_3_9',
                           'MA_3_12', 'MOM_1', 'MOM_2', 'MOM_3', 'MOM_6', 'MOM_9',
                           'MOM_12']].values[1:]
technical.shape

#
predictor_matrix = np.concatenate([predictor_raw['yyyymm'][1:].values.reshape(-1, 1), log_equity_premium.reshape(-1, 1),
                     equity_premium.reshape(-1, 1), macro, technical], axis=1)

#
result_predictor = pd.DataFrame(predictor_matrix,
                                columns=['month', 'log_equity_premium', 'equity_premium', 'DP', 'DY', 'EP', 'SVAR',
                                         'BM', 'NTIS', 'TBL', 'LTR', 'TMS', 'DFY', 'DFR', 'INFL','MA_1_9', 'MA_1_12',
                                         'MA_2_9', 'MA_2_12', 'MA_3_9', 'MA_3_12', 'MOM_1', 'MOM_2', 'MOM_3', 'MOM_6',
                                         'MOM_9', 'MOM_12'])


result_predictor.to_csv('result_predictor.csv', index=False)
