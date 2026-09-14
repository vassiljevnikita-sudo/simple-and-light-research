param(
  [Parameter(Mandatory=$true)][string]$V5Predictions,
  [string]$DailyStoreRoot = "artifacts\daily-parquet",
  [string]$OutputRoot = "artifacts\prediction-hold-qbd-surface",
  [ValidateRange(1, 30)][int]$PredictionMin = 1,
  [ValidateRange(1, 30)][int]$PredictionMax = 30,
  [ValidateRange(1, 30)][int]$HoldMax = 30,
  [int]$StageABudget = 48,
  [ValidateRange(1, 16)][int]$MaxWorkers = 16,
  [ValidateRange(1, 8)][int]$CoordinatorThreads = 8,
  [ValidateRange(1, 20)][int]$MinimumPlateauCells = 3,
  [switch]$FineGrainedFragmentCache,
  [switch]$Force,
  [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

if (-not (Test-Path $V5Predictions)) { throw "QBD prediction parquet missing: $V5Predictions" }
if (-not (Test-Path $DailyStoreRoot)) { throw "QBD daily store missing: $DailyStoreRoot" }
if ($PredictionMax -lt $PredictionMin) { throw "PredictionMax must be >= PredictionMin" }

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OPPORTUNITY_WINDOW_PIPELINE = "$CoordinatorThreads"
$env:OPPORTUNITY_TELEMETRY_PATH = (Join-Path $OutputRoot "qbd_multicore_telemetry.jsonl")

Write-Host "[preflight] Prediction x Hold QbD self-test"
python -m stock_predictor.backtests.opportunity_portfolio_research.prediction_hold_qbd_self_test
if ($LASTEXITCODE -ne 0) { throw "Prediction x Hold QbD self-test failed" }

Write-Host "[preflight] QbD throughput self-test"
python -m stock_predictor.backtests.opportunity_portfolio_research.prediction_hold_qbd_runner --self-test
if ($LASTEXITCODE -ne 0) { throw "Prediction x Hold QbD throughput self-test failed" }

$ArgsList = @(
  "-m", "stock_predictor.backtests.opportunity_portfolio_research.prediction_hold_qbd_runner",
  "--v5-predictions", $V5Predictions,
  "--daily-store-root", $DailyStoreRoot,
  "--output-root", $OutputRoot,
  "--prediction-min", "$PredictionMin",
  "--prediction-max", "$PredictionMax",
  "--hold-max", "$HoldMax",
  "--stage-a-budget", "$StageABudget",
  "--max-workers", "$MaxWorkers",
  "--coordinator-threads", "$CoordinatorThreads",
  "--minimum-plateau-cells", "$MinimumPlateauCells"
)
if ($FineGrainedFragmentCache) { $ArgsList += "--fine-grained-fragment-cache" }
if ($Force) { $ArgsList += "--force" }
if ($StopOnError) { $ArgsList += "--stop-on-error" }

Write-Host "[qbd] Starting explicit Prediction x Hold QbD runner. V4.5 exit overlay is NOT invoked."
Write-Host "[qbd] Throughput Stage 1: $CoordinatorThreads coordinators feeding one shared replay queue."
Write-Host "[qbd] Throughput Stage 2: $MaxWorkers replay workers."
if ($FineGrainedFragmentCache) {
  Write-Host "[qbd] Fine-grained fragment cache: ON (mid-cell fragment resume)"
} else {
  Write-Host "[qbd] Fine-grained fragment cache: OFF (cell-level resume / throughput mode)"
}
& python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Prediction x Hold QbD surface incomplete or failed" }

$SummaryPath = Join-Path $OutputRoot "qbd_summary.json"
if (-not (Test-Path $SummaryPath)) { throw "QBD summary missing: $SummaryPath" }
$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
if (-not $Summary.qbd_complete) { throw "QBD surface did not complete all requested measured cells" }

Write-Output "PREDICTION_HOLD_QBD_SURFACE_COMPLETE"
Write-Output "Prediction range: H$PredictionMin-H$PredictionMax"
Write-Output "Hold constraint: D <= min(H, $HoldMax)"
Write-Output "Entry grid per cell: 4 quantiles x 3 top fractions x 4 max_names = 48"
Write-Output "Runner: prediction_hold_qbd_runner.py"
Write-Output "Replay workers: $MaxWorkers"
Write-Output "Window coordinators: $CoordinatorThreads"
Write-Output "Fine-grained fragment cache: $(if ($FineGrainedFragmentCache) { 'ON' } else { 'OFF' })"
Write-Output "V4.5 exit overlay: NOT INVOKED"
Write-Output "Final holdout: CLOSED"
Write-Output "Surface artifacts: $OutputRoot"