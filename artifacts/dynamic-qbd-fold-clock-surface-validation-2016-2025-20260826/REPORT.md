# Dynamic-QBD Fold-Clock Surface Validation 2016-2025 Validation

Status: **COMPLETE**
Primary Development validation gate: **FAIL**
Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`; final holdout remains closed.

## Scope

- Families: 5400 across H1-H30, D1-D_H, N1-N6, FIXED and LEARNED_EXIT.
- A: initial entry model + calibration frozen.
- F: same entry recipe refit only on real matured-fold expansion; frozen between events.
- M: same entry recipe refit monthly.
- Learned-exit models are initial-causal and frozen/matched across A/F/M so the contrast isolates entry-refit frequency.

## Predeclared gates

- F_MINUS_A / ALL: **FAIL**, median plateau CAGR-excess delta +1.1104%.
- F_MINUS_A / FIXED: **FAIL**, median plateau CAGR-excess delta +1.9728%.
- F_MINUS_A / LEARNED_EXIT: **FAIL**, median plateau CAGR-excess delta +0.8203%.
- M_MINUS_F / ALL: **PASS**, median plateau CAGR-excess delta -3.6502%.
- M_MINUS_F / FIXED: **PASS**, median plateau CAGR-excess delta -4.4039%.
- M_MINUS_F / LEARNED_EXIT: **PASS**, median plateau CAGR-excess delta -3.1849%.

## Execution

- Worker threads: 26 with one native numerical thread each.
- Replay topology: 24 isolated horizon processes, 1 family workers per horizon; next horizon is queued on completion.
- CPU capacity target: 80%; enforced capacity fraction 81.25%.
- Aggregate Windows memory ceiling: 90.0 GiB; soft target 84.0 GiB.
- NWinfo telemetry is diagnostic-only and fail-open.

No gate in this report grants promotion or capital authority.
