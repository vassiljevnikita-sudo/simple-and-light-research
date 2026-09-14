param(
  [Parameter(Mandatory=$true)][string]$V5Predictions,
  [Parameter(Mandatory=$true)][string]$DailyStoreRoot,
  [Parameter(Mandatory=$true)][string]$LearnedExitPredictions,
  [string]$OutputRoot = "artifacts/learned-exit-qbd-profit",
  [string]$MaxNames = "1,2,3,4,5,6",
  [int]$MaxWorkers = 8,
  [double]$TaxAllowanceEur = 1000.0,
  [double]$BenchmarkPartialExemption = 0.30,
  [switch]$Force,
  [switch]$StopOnError
)
$ErrorActionPreference = "Stop"
$args = @(
  "-m", "stock_predictor.backtests.opportunity_portfolio_research.learned_exit_qbd_suite",
  "--v5-predictions", $V5Predictions,
  "--daily-store-root", $DailyStoreRoot,
  "--learned-exit-predictions", $LearnedExitPredictions,
  "--output-root", $OutputRoot,
  "--max-names", $MaxNames,
  "--max-workers", "$MaxWorkers",
  "--tax-allowance-eur", "$TaxAllowanceEur",
  "--benchmark-partial-exemption", "$BenchmarkPartialExemption"
)
if ($Force) { $args += "--force" }
if ($StopOnError) { $args += "--stop-on-error" }
python @args
exit $LASTEXITCODE
