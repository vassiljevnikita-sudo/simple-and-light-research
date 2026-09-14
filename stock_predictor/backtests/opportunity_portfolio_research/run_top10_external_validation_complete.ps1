[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
throw @"
SUPERSEDED_CAUSAL_RETRAINING_LAUNCHER

This launcher would build new causal prediction models and therefore does not
test the frozen Top-10 models. Use run_top10_frozen_live_replay.ps1 instead.
That runner loads existing frozen model artifacts, produces feature-only live
inference, and replays frozen policies without retraining or re-optimisation.
"@
