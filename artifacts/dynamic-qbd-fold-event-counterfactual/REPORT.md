# Dynamic-QBD Fold-Event Counterfactual

Status: **COMPLETE**
Authority: SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT; final holdout remains closed.

## Contract

- NEW and OLD branch from the exact same pre-event portfolio state.
- NEW follows the actual Fold-Clock generation and becomes canonical for the next event.
- OLD keeps the prior model and prior calibration only for the current Fold interval.
- Entry recipe is identical; only model generation/calibration age differs.
- FIXED exits only; no learned-exit coverage confound.
- No activation threshold or selector is fitted by this suite.

## Aggregate

- Non-initial Fold events: 180.
- Fixed families: 2790.
- Event-family rows: 16740.
- Positive median event fraction: 25.56%.
- Return-and-risk helpful fraction: 10.00%.
- Return-and-risk harmful fraction: 10.56%.

## Most harmful horizons

- H08: -5.6482%; positive events 0.0%.
- H19: -4.7857%; positive events 0.0%.
- H07: -4.4259%; positive events 0.0%.
- H16: -2.1774%; positive events 0.0%.
- H09: -1.4222%; positive events 0.0%.

## Most helpful horizons

- H11: +8.3496%; positive events 66.7%.
- H13: +5.1109%; positive events 66.7%.
- H02: +2.6060%; positive events 50.0%.
- H12: +2.0057%; positive events 50.0%.
- H29: +1.3494%; positive events 50.0%.

## Weakest event years

- 2023: -4.8815%; positive events 28.3%.
- 2021: +0.0000%; positive events 11.7%.
- 2022: +0.0000%; positive events 36.7%.

Post-event signal-shift fields use future interval behavior and are diagnostic only. They are forbidden as activation inputs.

No result in this report grants promotion or capital authority.
