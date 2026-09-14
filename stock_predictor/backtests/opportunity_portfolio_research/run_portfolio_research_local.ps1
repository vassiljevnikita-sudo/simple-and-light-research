param(
  [string]$V5Predictions = "artifacts\v5-local\training\signal\selected-walk-forward-predictions.parquet",
  [string]$DailyStoreRoot = "artifacts\daily-parquet",
  [string]$OutputRoot = "artifacts\opportunity-portfolio-research",
  [int]$StageABudget = 48,
  [ValidateRange(1, 12)]
  [int]$MaxWorkers = 12,
  [ValidateRange(1, 4)]
  [int]$CoordinatorThreads = 4,
  [int]$NativeThreadsPerWorker = 1,
  [ValidateSet("process", "thread")]
  [string]$ParallelBackend = "process",
  [string]$FragmentCachePath = "",
  [switch]$NoFragmentReuse,
  [switch]$ClearFragmentCache,
  [switch]$OpenFinalHoldout
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot
if ($MaxWorkers -lt 1) { throw "MaxWorkers must be >= 1" }
if ($NativeThreadsPerWorker -lt 1) { throw "NativeThreadsPerWorker must be >= 1" }

# Ryzen 7 5800X target: 16 hardware threads / logical CPUs. The process backend
# normally uses 12 long-lived replay processes on 12 logical CPUs. Four additional
# logical CPUs are left out of the replay affinity map and host the coordinator queue,
# SQLite/IPC, Windows and other short-lived work. Independent outer windows are fed
# through CoordinatorThreads lightweight coordinator threads into the same replay
# ProcessPool. Native numerical libraries stay single-threaded inside each process.
$env:OMP_NUM_THREADS = "$NativeThreadsPerWorker"
$env:OPENBLAS_NUM_THREADS = "$NativeThreadsPerWorker"
$env:MKL_NUM_THREADS = "$NativeThreadsPerWorker"
$env:NUMEXPR_NUM_THREADS = "$NativeThreadsPerWorker"
$env:BLIS_NUM_THREADS = "$NativeThreadsPerWorker"
$env:PYTHONUNBUFFERED = "1"
$env:OPPORTUNITY_WINDOW_PIPELINE = "$CoordinatorThreads"

# Persist deterministic search fragments so Ctrl+C/restarts reuse completed work.
# The multicore backend and window pipeline are execution-only and intentionally
# preserve the existing research fragment namespace.
if (-not $NoFragmentReuse) {
  if ([string]::IsNullOrWhiteSpace($FragmentCachePath)) {
    $FragmentCachePath = Join-Path $OutputRoot ".fragments\replay_cache.sqlite3"
  }
  if (-not [System.IO.Path]::IsPathRooted($FragmentCachePath)) {
    $FragmentCachePath = Join-Path $RepoRoot $FragmentCachePath
  }

  $FinalReplayCachePath = Join-Path (Split-Path $FragmentCachePath -Parent) "final_replay_cache.sqlite3"
  if ($ClearFragmentCache) {
    foreach ($cacheFile in @($FragmentCachePath, $FinalReplayCachePath)) {
      Remove-Item -Force -ErrorAction SilentlyContinue $cacheFile
      Remove-Item -Force -ErrorAction SilentlyContinue "$cacheFile-wal"
      Remove-Item -Force -ErrorAction SilentlyContinue "$cacheFile-shm"
    }
    Write-Host "Persistent replay caches cleared explicitly."
  }
  $env:OPPORTUNITY_FRAGMENT_CACHE_PATH = $FragmentCachePath
  Write-Host "Fragment reuse: ON  ($FragmentCachePath)"
  if ($ParallelBackend -eq "process") {
    Write-Host "Full replay reuse: ON  ($FinalReplayCachePath)"
  }
} else {
  Remove-Item Env:OPPORTUNITY_FRAGMENT_CACHE_PATH -ErrorAction SilentlyContinue
  Write-Host "Fragment reuse: OFF"
}

if ($ParallelBackend -eq "process") {
  $telemetryPath = Join-Path $OutputRoot "multicore_telemetry.jsonl"
  if (-not [System.IO.Path]::IsPathRooted($telemetryPath)) { $telemetryPath = Join-Path $RepoRoot $telemetryPath }
  $env:OPPORTUNITY_TELEMETRY_PATH = $telemetryPath
  $EntryModule = "stock_predictor.backtests.opportunity_portfolio_research.portfolio_research_portfolio_research_multicore_entrypoint"
  Write-Host "Parallel backend: PROCESS  replay_workers=$MaxWorkers coordinator_threads=$CoordinatorThreads native_threads=$NativeThreadsPerWorker"
} else {
  $EntryModule = "stock_predictor.backtests.opportunity_portfolio_research.portfolio_research_cli"
  Write-Host "Parallel backend: THREAD  requested_workers=$MaxWorkers native_threads=$NativeThreadsPerWorker"
}

$argsList = @(
  "-m", $EntryModule,
  "--v5-predictions", $V5Predictions,
  "--daily-store-root", $DailyStoreRoot,
  "--output-root", $OutputRoot,
  "--stage-a-budget", "$StageABudget",
  "--max-workers", "$MaxWorkers"
)
if ($OpenFinalHoldout) { $argsList += "--open-final-holdout" }
python @argsList
exit $LASTEXITCODE
