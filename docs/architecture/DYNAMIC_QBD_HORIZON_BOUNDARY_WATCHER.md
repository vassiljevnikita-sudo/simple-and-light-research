# Dynamic-QBD Horizon Boundary Watcher

The full-space runner can change execution architecture only at a completed
horizon checkpoint. `dynamic_qbd_horizon_boundary_watcher.ps1` provides that
handoff without waiting for all H1-H30 work to finish.

The watcher considers a horizon complete only when its family parquet, yearly
parquet, and JSON metadata exist, the metadata has `complete=true`, and all
three files are stable across a second observation. It therefore cannot switch
on a partially written horizon.

The old parent is validated by its command line before the process tree is
stopped. The new command is started from the same repository working directory
with the supplied argument array. The switch is explicit: use
`-StopParentAtBoundary` only when the resumed command has the intended new
execution semantics. The watcher never opens the final holdout or changes
research semantics; it changes scheduling/resume execution only.

Example (PowerShell; replace placeholders):

```powershell
$resume = @(
  "-m",
  "stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_full_space_fold_clock_validation",
  "--signal-panel", "<signal-panel>",
  "--learned-exit-candidate-metrics", "<exit-metrics>",
  "--daily-store-root", "<daily-store-root>",
  "--output-root", "<same-output-root>",
  "--horizon-workers", "24",
  "--code-commit", "<new-code-commit>"
)

& .\stock_predictor\backtests\opportunity_portfolio_research\dynamic_qbd_horizon_boundary_watcher.ps1 `
  -ParentPid <old-parent-pid> `
  -OutputRoot "<same-output-root>" `
  -BoundaryHorizons 7 `
  -ResumeArguments $resume `
  -StopParentAtBoundary
```

Use multiple values in `-BoundaryHorizons` when the switch must wait for a
specific checkpoint set, for example `-BoundaryHorizons 7,8,9`. The watcher
writes `horizon-boundary-watcher.jsonl` and redirects the resumed process to
`horizon-watcher-resume.stdout.log` / `.stderr.log`. These runtime artifacts
are local run outputs, not source-controlled evidence.
