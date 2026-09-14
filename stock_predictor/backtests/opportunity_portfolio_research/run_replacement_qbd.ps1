param(
  [Parameter(Mandatory=$true)][string]$V5Predictions,
  [string]$DailyStoreRoot = "artifacts\daily-parquet",
  [string]$Phase1DesignSpace = "artifacts\prediction-hold-qbd-surface\qbd_design_space.csv",
  [string]$Phase2Summary = "artifacts\allocation-qbd\allocation_qbd_summary.json",
  [string]$Phase2TreatmentSummary = "artifacts\allocation-qbd\allocation_qbd_treatment_summary.csv",
  [string]$OutputRoot = "artifacts\replacement-qbd",
  [string]$Treatments = "",
  [int]$StageABudget = 48,
  [ValidateRange(1, 12)][int]$MaxWorkers = 12,
  [ValidateRange(1, 8)][int]$CoordinatorThreads = 4,
  [switch]$FineGrainedFragmentCache,
  [switch]$Force,
  [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

if (-not (Test-Path $V5Predictions)) { throw "Replacement QbD prediction parquet missing: $V5Predictions" }
if (-not (Test-Path $DailyStoreRoot)) { throw "Replacement QbD daily store missing: $DailyStoreRoot" }
if (-not (Test-Path $Phase1DesignSpace)) { throw "Phase-1 QbD design space missing: $Phase1DesignSpace" }
if (-not (Test-Path $Phase2Summary)) { throw "Phase-2 Allocation QbD summary missing: $Phase2Summary" }
if (-not (Test-Path $Phase2TreatmentSummary)) { throw "Phase-2 Allocation QbD treatment summary missing: $Phase2TreatmentSummary" }

$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
$env:BLIS_NUM_THREADS="1"
$env:PYTHONUNBUFFERED="1"
$env:OPPORTUNITY_WINDOW_PIPELINE="$CoordinatorThreads"
$env:OPPORTUNITY_TELEMETRY_PATH=(Join-Path $OutputRoot "replacement_qbd_multicore_telemetry.jsonl")

Write-Host "[preflight] Phase-3 Replacement QbD self-test"
python -m stock_predictor.backtests.opportunity_portfolio_research.replacement_qbd_self_test
if ($LASTEXITCODE -ne 0) { throw "Phase-3 Replacement QbD self-test failed" }

$ArgsList=@(
  "-m","stock_predictor.backtests.opportunity_portfolio_research.replacement_qbd_runner",
  "--v5-predictions",$V5Predictions,
  "--daily-store-root",$DailyStoreRoot,
  "--phase1-design-space",$Phase1DesignSpace,
  "--phase2-summary",$Phase2Summary,
  "--phase2-treatment-summary",$Phase2TreatmentSummary,
  "--output-root",$OutputRoot,
  "--stage-a-budget","$StageABudget",
  "--max-workers","$MaxWorkers",
  "--coordinator-threads","$CoordinatorThreads"
)
if ($Treatments) { $ArgsList += @("--treatments",$Treatments) }
if ($FineGrainedFragmentCache) { $ArgsList += "--fine-grained-fragment-cache" }
if ($Force) { $ArgsList += "--force" }
if ($StopOnError) { $ArgsList += "--stop-on-error" }

Write-Host "[replacement-qbd] Starting explicit Phase-3 runner. H/D frozen from Phase-1; allocation=EQUAL_ACTIVE frozen from Phase-2; sleeve=0.50; V4.5 NOT invoked; final holdout CLOSED."
& python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Phase-3 Replacement QbD surface incomplete or failed" }

$SummaryPath=Join-Path $OutputRoot "replacement_qbd_summary.json"
if (-not (Test-Path $SummaryPath)) { throw "Replacement QbD summary missing: $SummaryPath" }
$Summary=Get-Content $SummaryPath -Raw | ConvertFrom-Json
if (-not $Summary.qbd_complete) { throw "Replacement QbD did not complete all requested measured cells" }

Write-Output "REPLACEMENT_QBD_SURFACE_COMPLETE"
Write-Output "Entry grid per cell: 48"
Write-Output "Default treatments: 2"
Write-Output "Default surface cells: 28"
Write-Output "Frozen allocation: EQUAL_ACTIVE"
Write-Output "Runner: replacement_qbd_runner.py"
Write-Output "V4.5 exit overlay: NOT INVOKED"
Write-Output "Final holdout: CLOSED"
Write-Output "Surface artifacts: $OutputRoot"
