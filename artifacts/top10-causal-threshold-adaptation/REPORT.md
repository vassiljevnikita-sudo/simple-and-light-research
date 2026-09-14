# Top-10 Causal Threshold Adaptation

Status: **COMPLETE**

Three weighting profiles were predeclared and run in parallel. This suite does not promote a winner from forward performance.

- W_RECENT = 0.50 / 0.30 / 0.20 for 1M / 3M / 6M
- W_MID = 0.20 / 0.50 / 0.30
- W_LONG = 0.20 / 0.30 / 0.50

Outcome evidence uses only fully matured rows at each assessment. Any change becomes effective on the next decision session.
The replacement threshold can only relax, never tighten, the original daily threshold.

## Results

| model_id | profile | w_1m | w_3m | w_6m | trade_count | cagr_excess | terminal_wealth_excess_eur |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01_L_H11_D03_N1 | W_LONG | 0.2 | 0.3 | 0.5 | 40 | -1.5738% | -605.413178272669 |
| R02_F_H24_D05_N1 | W_LONG | 0.2 | 0.3 | 0.5 | 32 | -13.7755% | -4768.121990843989 |
| R03_L_H28_D21_N1 | W_LONG | 0.2 | 0.3 | 0.5 | 20 | -0.8209% | -317.7850119568284 |
| R04_L_H28_D21_N5 | W_LONG | 0.2 | 0.3 | 0.5 | 30 | 13.2562% | 5768.442480649632 |
| R05_L_H28_D21_N4 | W_LONG | 0.2 | 0.3 | 0.5 | 30 | 13.2562% | 5768.442480649632 |
| R06_L_H28_D21_N6 | W_LONG | 0.2 | 0.3 | 0.5 | 30 | 13.2562% | 5768.442480649632 |
| R07_L_H28_D21_N2 | W_LONG | 0.2 | 0.3 | 0.5 | 26 | 10.9956% | 4697.2986304461265 |
| R08_L_H28_D21_N3 | W_LONG | 0.2 | 0.3 | 0.5 | 29 | 13.2295% | 5755.609504301356 |
| R09_L_H24_D21_N5 | W_LONG | 0.2 | 0.3 | 0.5 | 28 | 14.1697% | 6211.856312951913 |
| R10_L_H24_D21_N6 | W_LONG | 0.2 | 0.3 | 0.5 | 28 | 14.1697% | 6211.856312951913 |
| R01_L_H11_D03_N1 | W_MID | 0.2 | 0.5 | 0.3 | 40 | -1.5738% | -605.413178272669 |
| R02_F_H24_D05_N1 | W_MID | 0.2 | 0.5 | 0.3 | 31 | -5.0769% | -1895.3683594160084 |
| R03_L_H28_D21_N1 | W_MID | 0.2 | 0.5 | 0.3 | 19 | 1.6464% | 650.8166268587593 |
| R04_L_H28_D21_N5 | W_MID | 0.2 | 0.5 | 0.3 | 27 | 11.4419% | 4905.866338130749 |
| R05_L_H28_D21_N4 | W_MID | 0.2 | 0.5 | 0.3 | 27 | 11.4419% | 4905.866338130749 |
| R06_L_H28_D21_N6 | W_MID | 0.2 | 0.5 | 0.3 | 27 | 11.4419% | 4905.866338130749 |
| R07_L_H28_D21_N2 | W_MID | 0.2 | 0.5 | 0.3 | 24 | 10.4731% | 4455.015934652736 |
| R08_L_H28_D21_N3 | W_MID | 0.2 | 0.5 | 0.3 | 27 | 11.4419% | 4905.866338130749 |
| R09_L_H24_D21_N5 | W_MID | 0.2 | 0.5 | 0.3 | 25 | -0.3596% | -139.7482485054952 |
| R10_L_H24_D21_N6 | W_MID | 0.2 | 0.5 | 0.3 | 25 | -0.3596% | -139.7482485054952 |
| R01_L_H11_D03_N1 | W_RECENT | 0.5 | 0.3 | 0.2 | 45 | 2.5324% | 1008.5434358707716 |
| R02_F_H24_D05_N1 | W_RECENT | 0.5 | 0.3 | 0.2 | 32 | 5.2839% | 2153.286354740987 |
| R03_L_H28_D21_N1 | W_RECENT | 0.5 | 0.3 | 0.2 | 18 | -5.8438% | -2167.317538800944 |
| R04_L_H28_D21_N5 | W_RECENT | 0.5 | 0.3 | 0.2 | 23 | -3.5254% | -1333.7565123728273 |
| R05_L_H28_D21_N4 | W_RECENT | 0.5 | 0.3 | 0.2 | 23 | -3.5254% | -1333.7565123728273 |
| R06_L_H28_D21_N6 | W_RECENT | 0.5 | 0.3 | 0.2 | 23 | -3.5254% | -1333.7565123728273 |
| R07_L_H28_D21_N2 | W_RECENT | 0.5 | 0.3 | 0.2 | 22 | -3.6533% | -1380.6360794835127 |
| R08_L_H28_D21_N3 | W_RECENT | 0.5 | 0.3 | 0.2 | 23 | -3.5254% | -1333.7565123728273 |
| R09_L_H24_D21_N5 | W_RECENT | 0.5 | 0.3 | 0.2 | 26 | 9.2201% | 3881.844847911485 |
| R10_L_H24_D21_N6 | W_RECENT | 0.5 | 0.3 | 0.2 | 26 | 9.2201% | 3881.844847911485 |

No models were trained, no predictions were regenerated, no threshold or weight profile was optimized, and the final holdout remained closed.
