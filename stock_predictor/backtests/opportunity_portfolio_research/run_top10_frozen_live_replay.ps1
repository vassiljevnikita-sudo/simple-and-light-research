param(
  [Parameter(Mandatory=$true)][string]$PanelRoot,
  [Parameter(Mandatory=$true)][string]$TrainingRoot,
  [Parameter(Mandatory=$true)][string]$FeatureBuilderSource,
  [Parameter(Mandatory=$true)][string]$KnownProfitRoot,
  [Parameter(Mandatory=$true)][string]$DailyStoreRoot,
  [string]$OutputRoot = "artifacts\top10-frozen-live-replay",
  [double]$InitialCapital = 10000.0,
  [double]$TaxAllowanceEur = 1000.0,
  [double]$BenchmarkPartialExemption = 0.30
)

$ErrorActionPreference = "Stop"
& python -m stock_predictor.backtests.opportunity_portfolio_research.top10_frozen_live_replay `
  --panel-root $PanelRoot `
  --training-root $TrainingRoot `
  --feature-builder-source $FeatureBuilderSource `
  --known-profit-root $KnownProfitRoot `
  --daily-store-root $DailyStoreRoot `
  --output-root $OutputRoot `
  --initial-capital $InitialCapital `
  --tax-allowance-eur $TaxAllowanceEur `
  --benchmark-partial-exemption $BenchmarkPartialExemption
exit $LASTEXITCODE
