# Top-10 Model-Specific Memory Controller V3

Contract: `TOP10_MODEL_SPECIFIC_MEMORY_CONTROLLER_V3`

## Purpose

V3 tests complete `model + memory-controller` systems instead of forcing one universal memory controller across all Top-10 policies.

The prior classes are frozen from the already completed fixed-profile development experiment:

- `R01_L_H11_D03_N1` -> `SHORT` (`W_RECENT`)
- `R02_F_H24_D05_N1` -> `SHORT` (`W_RECENT`)
- `R03_L_H28_D21_N1` -> `MID` (`W_MID`)
- `R04_L_H28_D21_N5` -> `LONG` (`W_LONG`)
- `R05_L_H28_D21_N4` -> `LONG` (`W_LONG`)
- `R06_L_H28_D21_N6` -> `LONG` (`W_LONG`)
- `R07_L_H28_D21_N2` -> `LONG` (`W_LONG`)
- `R08_L_H28_D21_N3` -> `LONG` (`W_LONG`)
- `R09_L_H24_D21_N5` -> `LONG` (`W_LONG`)
- `R10_L_H24_D21_N6` -> `LONG` (`W_LONG`)

The suite hard-gates this mapping against the committed `causal_threshold_adaptation_results.csv`. It is not recomputed or switched online.

## Development-reuse status

The prior classes were selected after inspection of the existing 2023-2026 fixed-profile results. Therefore this suite is explicitly:

`DEVELOPMENT_REUSE_NOT_INDEPENDENT_OOS`

The replay may be used to design/freeze complete systems before Final Holdout, but it is not an independent OOS proof. Final Holdout remains closed.

## Twelve lag experts

Each model has its own M1...M12 controller state:

- `M1`: latest 21 fully matured decision sessions
- `M2`: previous 21 sessions
- ...
- `M12`: oldest 21-session block in the 252-session memory

These are age slots, not persistent historical cohorts. This deliberately returns to the lag-slot semantics after Cohort V2 showed that attaching weight to specific historical cohorts was harmful.

## Prior shapes

The old profile masses are mapped onto non-overlapping age bands:

- short band: `M1-M3`
- mid band: `M4-M6`
- long band: `M7-M12`

Within each band, mass is uniform.

### SHORT

`0.50 / 0.30 / 0.20`

- M1-M3: 16.6667% each
- M4-M6: 10% each
- M7-M12: 3.3333% each

### MID

`0.20 / 0.50 / 0.30`

- M1-M3: 6.6667% each
- M4-M6: 16.6667% each
- M7-M12: 5% each

### LONG

`0.20 / 0.30 / 0.50`

- M1-M3: 6.6667% each
- M4-M6: 10% each
- M7-M12: 8.3333% each

No alternative monthly prior shapes are searched.

## Online controller

Each model owns an independent state vector. At every 21-session assessment:

1. Only outcomes with `terminal_date <= assessment_date` are visible.
2. Relative candidates are defined using that model's own horizon, `top_fraction`, and `max_names`.
3. M1...M12 receive the same matured performance-quality signal as Monthly Controller V1.
4. Cross-lag performance is converted to the inherited rank signal.
5. The target is anchored to the frozen model prior:

`target_i ∝ prior_i * exp(HEDGE_ETA * signal_i)`

6. Current weights move toward that target under the inherited V1 safety limits.
7. The new state becomes effective strictly after the assessment date.

The model prior is reapplied on every update. Past performance can tilt the model's memory, but it cannot silently turn every model into the same universal controller.

## Safety limits

V3 inherits V1 limits to isolate the model-specific-prior change:

- cadence: 21 sessions
- weight floor: 2%
- weight ceiling: 25%
- max change: +/-2 percentage points per assessment
- Hedge eta: 1.0
- first update requires all twelve matured lag slots

No new speed/floor/ceiling/eta grid is optimized.

## Threshold arms

Each model is replayed under four arms:

1. `RAW`
2. `UNIFORM_12M`
3. `STATIC_MODEL_PRIOR`
4. `ADAPTIVE_MODEL_CONTROLLER`

Total: **40 portfolio replays**.

For the three adaptive/calibrated arms:

`replacement_threshold = historical_threshold_over_p99 * weighted_p99`

`effective_threshold = min(raw_threshold, replacement_threshold)`

The suite can only relax the original threshold, never tighten it.

## Time contract

### State preparation

`2020-09-01` through `2023-08-10`

Source: historical `WF_001 ... WF_007` OOS predictions. `WF_000` stays excluded.

Because the prior classes were defined later from development evidence, this is development-state reconstruction, not a claim that V3 was historically known in 2020.

### Development replay

`2023-08-11` through `2026-07-24`

Source: the existing causal-expanding entry/exit prediction artifacts (~17k checkpoints).

No models or predictions are regenerated.

### Final Holdout

Closed.

## Selection unit

The eventual selection unit is the complete system:

`entry model + H/D/N + exit policy + model-specific memory controller`

`development_model_system_comparison.csv` ranks systems for development analysis only and explicitly sets `selection_allowed = false`.

## Outputs

- `model_specific_memory_prior_manifest.csv`
- `model_specific_controller_state.csv`
- `model_specific_controller_state.parquet`
- `model_specific_controller_expert_evidence.parquet`
- `model_specific_live_thresholds.csv`
- `model_specific_live_thresholds.parquet`
- `model_specific_controller_results.csv`
- `model_specific_controller_trades.csv`
- `model_specific_controller_curves.parquet`
- `model_specific_weights_at_live_start.csv`
- `development_model_system_comparison.csv`
- `model_specific_controller_summary.json`
- `REPORT.md`

## Hard gates

The suite fails if the frozen prior mapping no longer matches the committed fixed-profile results, WF_000 enters warm-up, causal exit coverage is incomplete, a threshold uses the same/future assessment date, an adaptive threshold tightens RAW, learned-exit coverage drops below 100%, or Final Holdout is opened.
