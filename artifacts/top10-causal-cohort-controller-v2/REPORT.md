# Top-10 Causal Cohort Controller V2

Status: **COMPLETE**

**Development-reuse warning:** 2023-2026 was already inspected while designing V2; results are development evidence, not a fresh independent OOS claim.

V2 attaches weights to concrete 21-session cohorts, carries those weights as each cohort ages from M1 to M12, and uses magnitude-aware robust quality with confidence.

Hard controller limits: floor 1%, ceiling 35%, max +/-5 percentage points per assessment.

## V2 replay arms

| model_id | arm | trade_count | cagr_excess | terminal_wealth_excess_eur |
| --- | --- | --- | --- | --- |
| R01_L_H11_D03_N1 | COHORT_CONTROLLER_V2 | 70 | -6.8132% | -2505.8149731806243 |
| R02_F_H24_D05_N1 | COHORT_CONTROLLER_V2 | 60 | 0.5275% | 206.5500099937508 |
| R03_L_H28_D21_N1 | COHORT_CONTROLLER_V2 | 23 | -1.8185% | -698.0765252417223 |
| R04_L_H28_D21_N5 | COHORT_CONTROLLER_V2 | 43 | -16.7424% | -5644.994634395131 |
| R05_L_H28_D21_N4 | COHORT_CONTROLLER_V2 | 42 | -15.0104% | -5139.256992956789 |
| R06_L_H28_D21_N6 | COHORT_CONTROLLER_V2 | 43 | -16.7214% | -5638.957987434103 |
| R07_L_H28_D21_N2 | COHORT_CONTROLLER_V2 | 32 | -9.4226% | -3388.0635801926383 |
| R08_L_H28_D21_N3 | COHORT_CONTROLLER_V2 | 40 | -15.0910% | -5163.176669725766 |
| R09_L_H24_D21_N5 | COHORT_CONTROLLER_V2 | 42 | 18.6529% | 8477.91271214596 |
| R10_L_H24_D21_N6 | COHORT_CONTROLLER_V2 | 42 | 18.6176% | 8459.484378717258 |
| R01_L_H11_D03_N1 | COHORT_UNIFORM_12M | 72 | -6.0540% | -2241.2326251586546 |
| R02_F_H24_D05_N1 | COHORT_UNIFORM_12M | 55 | 13.6931% | 5979.787169535055 |
| R03_L_H28_D21_N1 | COHORT_UNIFORM_12M | 18 | 12.6743% | 5489.18092326499 |
| R04_L_H28_D21_N5 | COHORT_UNIFORM_12M | 29 | 16.6592% | 7451.631025619601 |
| R05_L_H28_D21_N4 | COHORT_UNIFORM_12M | 29 | 16.6584% | 7451.218345858899 |
| R06_L_H28_D21_N6 | COHORT_UNIFORM_12M | 29 | 16.6587% | 7451.334841164102 |
| R07_L_H28_D21_N2 | COHORT_UNIFORM_12M | 23 | 12.4818% | 5397.360326444632 |
| R08_L_H28_D21_N3 | COHORT_UNIFORM_12M | 26 | 12.6844% | 5494.025370394615 |
| R09_L_H24_D21_N5 | COHORT_UNIFORM_12M | 45 | 10.8907% | 4648.492636302057 |
| R10_L_H24_D21_N6 | COHORT_UNIFORM_12M | 45 | 10.8809% | 4643.945905083448 |
| R01_L_H11_D03_N1 | RAW | 3 | -1.6014% | -615.8689052095433 |
| R02_F_H24_D05_N1 | RAW | 0 | 0.0000% | 0.0 |
| R03_L_H28_D21_N1 | RAW | 0 | 0.0000% | 0.0 |
| R04_L_H28_D21_N5 | RAW | 0 | 0.0000% | 0.0 |
| R05_L_H28_D21_N4 | RAW | 0 | 0.0000% | 0.0 |
| R06_L_H28_D21_N6 | RAW | 0 | 0.0000% | 0.0 |
| R07_L_H28_D21_N2 | RAW | 0 | 0.0000% | 0.0 |
| R08_L_H28_D21_N3 | RAW | 0 | 0.0000% | 0.0 |
| R09_L_H24_D21_N5 | RAW | 0 | 0.0000% | 0.0 |
| R10_L_H24_D21_N6 | RAW | 0 | 0.0000% | 0.0 |

## V2 controller versus frozen V1 reference

| model_id | cagr_excess_v1 | cagr_excess_v2 | cagr_excess_delta_v2_minus_v1 |
| --- | --- | --- | --- |
| R01_L_H11_D03_N1 | -0.5282% | -6.8132% | -6.2849% |
| R02_F_H24_D05_N1 | 8.1589% | 0.5275% | -7.6314% |
| R03_L_H28_D21_N1 | 6.0333% | -1.8185% | -7.8518% |
| R04_L_H28_D21_N5 | 5.0333% | -16.7424% | -21.7757% |
| R05_L_H28_D21_N4 | 5.0436% | -15.0104% | -20.0540% |
| R06_L_H28_D21_N6 | 5.0680% | -16.7214% | -21.7894% |
| R07_L_H28_D21_N2 | 4.9417% | -9.4226% | -14.3643% |
| R08_L_H28_D21_N3 | 5.0133% | -15.0910% | -20.1043% |
| R09_L_H24_D21_N5 | 0.5157% | 18.6529% | 18.1371% |
| R10_L_H24_D21_N6 | 0.5098% | 18.6176% | 18.1078% |

No model was trained, no prediction was regenerated, no threshold/weight grid was searched, and the final holdout remained closed.
