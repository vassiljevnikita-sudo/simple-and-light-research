param(
  [Parameter(Mandatory=$true)]
  [string]$V5Predictions,
  [string]$DailyStoreRoot = "artifacts\daily-parquet",
  [string]$OutputRoot = "artifacts\opportunity-portfolio-h1-30",
  [int]$StageABudget = 48,
  [ValidateRange(1, 12)]
  [int]$MaxWorkers = 12,
  [ValidateRange(1, 4)]
  [int]$CoordinatorThreads = 4,
  [switch]$SkipClean
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

if ($MaxWorkers -ne 12) { Write-Warning "Research target is 12 replay workers; requested $MaxWorkers." }
if ($CoordinatorThreads -ne 4) { Write-Warning "Research target is 4 coordinator threads; requested $CoordinatorThreads." }
if (-not (Test-Path $V5Predictions)) { throw "H1-H30 prediction parquet missing: $V5Predictions" }

$CleanRoot = Join-Path $OutputRoot "clean"
$ExitRoot = Join-Path $OutputRoot "v45-exit"
$TelemetryPath = Join-Path $CleanRoot "multicore_telemetry.jsonl"

function Invoke-ResearchSelfTest {
  param(
    [Parameter(Mandatory=$true)][string]$Module,
    [string[]]$ModuleArgs = @()
  )
  Write-Host "[preflight] $Module $($ModuleArgs -join ' ')"
  python -m $Module @ModuleArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Preflight self-test failed: $Module"
  }
}

# Reuse the complete established Opportunity Portfolio regression suite before
# expanding it to H1-H30. These tests protect accounting, performance, persistent
# cache/restart behavior, CPU topology, 12+4 multicore dispatch, BrokenProcessPool
# recovery, runtime gates and post-selection evidence. The H1-H30-specific test is
# additive; it does not replace the existing suite.
Invoke-ResearchSelfTest `
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_research_cli" `
  @("--self-test")
$ExistingSelfTests = @(
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_policy_search_performance_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_replay_portfolio_replay_fragment_cache_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_portfolio_replay_result_cache_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.cpu_topology_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_replay_process_backend_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_research_multicore_pipeline_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_portfolio_resilient_process_pool_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_research_validation_runtime_self_test",
  "stock_predictor.backtests.opportunity_portfolio_research.portfolio_portfolio_evidence_expansion_self_test"
)
foreach ($Module in $ExistingSelfTests) {
  Invoke-ResearchSelfTest $Module
}
Invoke-ResearchSelfTest "stock_predictor.backtests.opportunity_portfolio_research.h1_30_matched_hold_suite_self_test"
Write-Host "[preflight] Established Opportunity Portfolio suite + H1-H30 extensions: PASS"

if (-not $SkipClean) {
  & (Join-Path $PSScriptRoot "run_h1_30_local.ps1") `
    -V5Predictions $V5Predictions `
    -DailyStoreRoot $DailyStoreRoot `
    -OutputRoot $CleanRoot `
    -StageABudget $StageABudget `
    -MaxWorkers $MaxWorkers `
    -CoordinatorThreads $CoordinatorThreads `
    -NativeThreadsPerWorker 1
  if ($LASTEXITCODE -ne 0) { throw "H1-H30 clean research failed" }
} else {
  Write-Host "Clean arm skipped explicitly; existing artifacts will be validated and reused."
}

$Outer = Join-Path $CleanRoot "outer_fold_results.csv"
$Consistency = Join-Path $CleanRoot "outer_replay_consistency.csv"
$TradeLog = Join-Path $CleanRoot "outer_oos_trade_log.csv"
if (-not (Test-Path $Outer)) { throw "H1_30_CLEAN_INPUT_MISSING: outer_fold_results.csv not found at $Outer. The Clean arm did not finish." }
if (-not (Test-Path $Consistency)) { throw "H1_30_CLEAN_INPUT_MISSING: outer_replay_consistency.csv not found at $Consistency. Post-selection evidence expansion did not finish." }
if (-not (Test-Path $TradeLog)) { throw "H1_30_CLEAN_INPUT_MISSING: outer_oos_trade_log.csv not found at $TradeLog. Run evidence_expansion against the existing Clean outer results before V4.5." }

python -c "import pandas as pd,sys; p=r'$Consistency'; d=pd.read_csv(p); col='consistent' if 'consistent' in d.columns else ('pass' if 'pass' in d.columns else None); ok=(not d.empty) and col is not None and d[col].astype(bool).all(); print('H1_30_CLEAN_REPLAY_CONSISTENCY', bool(ok), 'column=',col, 'rows=',len(d)); sys.exit(0 if ok else 4)"
if ($LASTEXITCODE -ne 0) { throw "Clean baseline consistency failed; V4.5 overlay will not run" }

python -c "import pandas as pd,sys; p=r'$Outer'; d=pd.read_csv(p); expected=set(range(1,31)); actual=set(d['horizon'].astype(int)); ok=actual==expected; print('H1_30_CLEAN_HORIZON_COVERAGE', bool(ok), 'missing=',sorted(expected-actual), 'unexpected=',sorted(actual-expected)); sys.exit(0 if ok else 5)"
if ($LASTEXITCODE -ne 0) { throw "Clean baseline must contain every horizon H1-H30; V4.5 overlay will not run" }

python -c "import pandas as pd,sys; outer=pd.read_csv(r'$Outer'); trades=pd.read_csv(r'$TradeLog'); expected=int(pd.to_numeric(outer['trade_count'],errors='coerce').fillna(0).sum()); actual=len(trades); ok=expected==actual; print('H1_30_CLEAN_TRADE_LOG_PARITY', bool(ok), 'expected=',expected, 'actual=',actual); sys.exit(0 if ok else 6)"
if ($LASTEXITCODE -ne 0) { throw "Clean OOS trade log does not match official outer-fold trade counts; V4.5 overlay will not run" }

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OPPORTUNITY_WINDOW_PIPELINE = "$CoordinatorThreads"
$env:OPPORTUNITY_TELEMETRY_PATH = $TelemetryPath

python -m stock_predictor.backtests.opportunity_portfolio_research.v45_exit_overlay `
  --v5-predictions $V5Predictions `
  --daily-store-root $DailyStoreRoot `
  --clean-output-root $CleanRoot `
  --output-root $ExitRoot `
  --max-workers $MaxWorkers `
  --coordinator-threads $CoordinatorThreads `
  --telemetry-path $TelemetryPath
if ($LASTEXITCODE -ne 0) { throw "H1-H30 V4.5 paired exit overlay failed" }

$ExitSummary = Join-Path $ExitRoot "v45_exit_summary.json"
if (-not (Test-Path $ExitSummary)) { throw "V4.5 exit summary missing" }

Write-Output "H1_30_TWO_ARM_DEVELOPMENT_SUITE_COMPLETE"
Write-Output "Arm A: H1-H30 clean, fixed holding_days=H"
Write-Output "Arm B: existing clean OOS entries + V4.5 early-exit candidates"
Write-Output "Replay workers: $MaxWorkers"
Write-Output "Coordinator threads: $CoordinatorThreads"
Write-Output "Native threads/worker: 1"
Write-Output "Telemetry: $TelemetryPath"
Write-Output "Final holdout: CLOSED"
Write-Output "Clean artifacts: $CleanRoot"
Write-Output "V4.5 exit artifacts: $ExitRoot"
