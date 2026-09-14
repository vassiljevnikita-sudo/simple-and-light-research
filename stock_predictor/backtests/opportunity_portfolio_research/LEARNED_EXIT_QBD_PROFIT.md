# Learned Exit H×D×N QbD Profit Suite

Contract: `LEARNED_EXIT_QBD_PROFIT_V1`

This suite treats QbD islands as a structured search/diagnostic surface, not as the production object. The final candidate is one concrete point. Primary selection is the **highest fold-compounded OOS after-tax terminal-wealth excess versus the after-tax URTH benchmark**; island median/connectivity are context.

## Frozen contract

- Entry horizon: H1..H30.
- Maximum holding: D1..D30 with `D <= H`.
- Concentration: N1..N6.
- Replacement: `IGNORE_NEW`.
- Allocation: `EQUAL_ACTIVE`.
- Sleeve: `0.50`.
- Inner entry grid only: 4 score quantiles × 3 top fractions = 12 candidates/cell.
- Primary roundtrip cost: 20 bps, identical for FIXED and LEARNED.
- Learned-exit source: commit `aa8f1b5be6c5c4947ac2446e06627c9c915bc7d9`, methodology `V5_E1_30_CONTINUATION_VALUE_PREQUENTIAL_V1`.
- Exit decision at session close; execution next session open.
- Exit threshold: predicted continuation excess `<= 0`.
- A D-session position consults `E(D-1)`, `E(D-2)`, ...; prediction-H is not reused as the exit horizon.
- If the required selected-WF continuation path is incomplete, the trade falls back to FIXED D, matching the earlier learned-exit replay contract.
- The suite consumes the selected walk-forward prediction parquet only. `DEVELOPMENT_FULL` models are not used for historical OOS decisions.
- Frozen validation remains closed and promotion remains false.

## Search and comparison

Learned surface: 465 valid H/D pairs × 6 max-names = **2,790 cells**.

FIXED reference: the existing 14-cell H23-H26/D3-D8 island, remeasured at N=1 under identical cost/tax accounting.

For FIXED and LEARNED the evaluator reports both median and maximum profit. It also reports positive connected Learned islands per N and ranks every point. The champion is the single point with the highest after-tax wealth excess, not the island with the best median.

## Tax assumptions

Defaults: 25% capital-gains tax, 5.5% solidarity surcharge, EUR 1,000 Sparer-Pauschbetrag, no church tax. The benchmark applies a configurable 30% partial-exemption scenario for a qualifying equity fund.

Benchmark tax remains an approximation because Vorabpauschale is not modeled. Outer-fold tax ledgers are independent, so the annual allowance can be reused across fold replays. This limitation is persisted in `run_summary.json`. Run an additional conservative ranking with `--tax-allowance-eur 0` if allowance-reset effects could change the champion.

## Inputs

The learned-exit input is the local selected-WF parquet from the completed E1-E30 run, e.g.:
`D:/simple-and-light-v5-h1-30-artifacts/training/e1-30-learned-exit-20260809/exit/selected-walk-forward-predictions.parquet`

## Outputs

- `cells/learned_exit/Hxx_Dxx_Nn.json`
- `cells/fixed/Hxx_Dxx_N1.json`
- `point_ranking.csv`
- `islands.json`
- `evaluation_summary.json`
- `run_summary.json`

The final holdout is intentionally not opened by this suite.
