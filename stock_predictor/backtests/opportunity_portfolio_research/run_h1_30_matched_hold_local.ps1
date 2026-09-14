param(
  [Parameter(Mandatory=$true)]
  [string]$V5Predictions,
  [string]$DailyStoreRoot = "artifacts\daily-parquet",
  [string]$OutputRoot = "artifacts\opportunity-portfolio-h1-30\clean",
  [int]$StageABudget = 48,
  [ValidateRange(1, 12)]
  [int]$MaxWorkers = 12,
  [ValidateRange(1, 4)]
  [int]$CoordinatorThreads = 4,
  [int]$NativeThreadsPerWorker = 1,
  [string]$FragmentCachePath = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

if ($MaxWorkers -lt 1) { throw "MaxWorkers must be >= 1" }
if ($NativeThreadsPerWorker -ne 1) { throw "H1-H30 research requires NativeThreadsPerWorker=1" }
if (-not (Test-Path $V5Predictions)) { throw "H1-H30 prediction parquet missing: $V5Predictions" }

# Keep the established opportunity-research execution architecture unchanged.
$env:OMP_NUM_THREADS = "$NativeThreadsPerWorker"
$env:OPENBLAS_NUM_THREADS = "$NativeThreadsPerWorker"
$env:MKL_NUM_THREADS = "$NativeThreadsPerWorker"
$env:NUMEXPR_NUM_THREADS = "$NativeThreadsPerWorker"
$env:BLIS_NUM_THREADS = "$NativeThreadsPerWorker"
$env:PYTHONUNBUFFERED = "1"
$env:OPPORTUNITY_WINDOW_PIPELINE = "$CoordinatorThreads"

if ([string]::IsNullOrWhiteSpace($FragmentCachePath)) {
  $FragmentCachePath = Join-Path $OutputRoot ".fragments\replay_cache.sqlite3"
}
if (-not [System.IO.Path]::IsPathRooted($FragmentCachePath)) {
  $FragmentCachePath = Join-Path $RepoRoot $FragmentCachePath
}
$env:OPPORTUNITY_FRAGMENT_CACHE_PATH = $FragmentCachePath

$telemetryPath = Join-Path $OutputRoot "multicore_telemetry.jsonl"
if (-not [System.IO.Path]::IsPathRooted($telemetryPath)) {
  $telemetryPath = Join-Path $RepoRoot $telemetryPath
}
$env:OPPORTUNITY_TELEMETRY_PATH = $telemetryPath

Write-Host "H1-H30 CLEAN adaptive Development"
Write-Host "Replay workers: $MaxWorkers"
Write-Host "Coordinator threads: $CoordinatorThreads"
Write-Host "Native threads/worker: $NativeThreadsPerWorker"
Write-Host "Telemetry: $telemetryPath"
Write-Host "Fragment cache: $FragmentCachePath"
Write-Host "Final holdout: CLOSED"

python -m stock_predictor.backtests.opportunity_portfolio_research.h1_30_matched_hold_entrypoint `
  --v5-predictions $V5Predictions `
  --daily-store-root $DailyStoreRoot `
  --output-root $OutputRoot `
  --stage-a-budget $StageABudget `
  --max-workers $MaxWorkers
exit $LASTEXITCODE
