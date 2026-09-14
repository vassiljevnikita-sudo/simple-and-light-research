# Frozen Top-10 Live Replay

Status: `FROZEN_MODEL_LIVE_REPLAY_COMPLETE`

This suite performed inference only. Every model binary was SHA-256 locked; no model was fitted, no entry policy was searched, and no threshold was recalibrated.

## Interpretation

- Historical regime backcast: 2016-01-01 to 2024-01-30 — diagnostic only; not causal OOS.
- True forward replay: 2024-01-31 to 2026-07-24 — separate fresh account, frozen model and policy, no retraining.
- Learned exits: every required decision is covered either by a frozen model prediction or by the explicit delisting contract; no alias or forward-fill resolution was used.
- Feature provenance: `PASSED_STRUCTURAL_CAUSAL_FEATURE_PROVENANCE_AUDIT`.
- Final holdout remains closed; point-in-time universe is not verified, so promotion remains blocked.

## Frozen identities (original order; not re-ranked)

| Rank | Model | Backcast CAGR-X | True-forward CAGR-X | Forward growth | Forward trades |
|---:|---|---:|---:|---:|---:|
| 1 | R01_L_H11_D03_N1 | 0.51% | -1.99% | 1.436x | 3 |
| 2 | R02_F_H24_D05_N1 | 1.97% | 0.00% | 1.498x | 0 |
| 3 | R03_L_H28_D21_N1 | 11.67% | -5.61% | 1.327x | 1 |
| 4 | R04_L_H28_D21_N5 | 15.16% | -5.61% | 1.327x | 1 |
| 5 | R05_L_H28_D21_N4 | 15.16% | -5.61% | 1.327x | 1 |
| 6 | R06_L_H28_D21_N6 | 15.16% | -5.61% | 1.327x | 1 |
| 7 | R07_L_H28_D21_N2 | 15.16% | -5.61% | 1.327x | 1 |
| 8 | R08_L_H28_D21_N3 | 15.16% | -5.61% | 1.327x | 1 |
| 9 | R09_L_H24_D21_N5 | 25.85% | -5.61% | 1.327x | 1 |
| 10 | R10_L_H24_D21_N6 | 25.85% | -5.61% | 1.327x | 1 |
