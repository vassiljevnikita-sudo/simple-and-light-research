param(
  [Parameter(Mandatory=$true)][string]$RawMinuteRoot,
  [string]$ArtifactRoot = "",
  [Parameter(Mandatory=$true)][string]$KnownProfitRoot,
  [Parameter(Mandatory=$true)][string]$DailyStoreRoot,
  [string]$V5Predictions = "",
  [string]$LearnedExitPredictions = "",
  [string]$StopExecutionPredictions = "",
  [string]$OutputRoot = "artifacts/top10-external-validation",
  [int]$EntryBudget = 12,
  [int]$MaxWorkers = 8,
  [double]$InitialCapital = 10000.0,
  [double]$TaxAllowanceEur = 1000.0,
  [double]$BenchmarkPartialExemption = 0.30
)

$ErrorActionPreference = "Stop"

function Resolve-CanonicalArtifact([string]$ExplicitPath, [string]$RelativePath, [string]$Label) {
  if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
    return $ExplicitPath
  }
  if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    throw "ArtifactRoot is required when $Label is not supplied explicitly."
  }
  return Join-Path $ArtifactRoot $RelativePath
}

$V5Predictions = Resolve-CanonicalArtifact `
  $V5Predictions `
  "training\signal\selected-walk-forward-predictions.parquet" `
  "V5Predictions"
$LearnedExitPredictions = Resolve-CanonicalArtifact `
  $LearnedExitPredictions `
  "training\e1-30-learned-exit-20260809\exit\selected-walk-forward-predictions.parquet" `
  "LearnedExitPredictions"
$StopExecutionPredictions = Resolve-CanonicalArtifact `
  $StopExecutionPredictions `
  "training\stop-execution\selected-walk-forward-predictions.parquet" `
  "StopExecutionPredictions"

$runnerArgs = @(
  "-m", "stock_predictor.backtests.opportunity_portfolio_research.top10_external_validation_strict",
  "--raw-minute-root", $RawMinuteRoot,
  "--v5-predictions", $V5Predictions,
  "--learned-exit-predictions", $LearnedExitPredictions,
  "--stop-execution-predictions", $StopExecutionPredictions,
  "--known-profit-root", $KnownProfitRoot,
  "--daily-store-root", $DailyStoreRoot,
  "--output-root", $OutputRoot,
  "--entry-budget", $EntryBudget,
  "--max-workers", $MaxWorkers,
  "--initial-capital", $InitialCapital,
  "--tax-allowance-eur", $TaxAllowanceEur,
  "--benchmark-partial-exemption", $BenchmarkPartialExemption
)
if (-not [string]::IsNullOrWhiteSpace($ArtifactRoot)) {
  $runnerArgs += @("--artifact-root", $ArtifactRoot)
}

& python @runnerArgs
exit $LASTEXITCODE
