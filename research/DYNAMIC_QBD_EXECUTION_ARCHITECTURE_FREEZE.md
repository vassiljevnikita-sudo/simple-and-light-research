# Dynamic-QBD Execution Architecture Freeze — Superseded

Status date: 2026-09-02

This decision has been superseded.

The prior statement that `v40.0.4.3` was a frozen final execution
architecture was too restrictive. More efficient architectures may be tested.

The current authority is:

[Dynamic-QBD Execution Minimum Baseline and Failure Ledger](DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md)

Current rule:

- `v40.0.4.3` is the **minimum acceptable baseline**, not the final design;
- future architectures may replace its mechanisms when they are measurably
  better;
- they must not reintroduce any already-documented historical failure;
- every historical failure is recorded as **failed because X**, together with
  the property that fixed it;
- old minor-version labels are not artificially assigned to defects when the
  repository no longer preserves a trustworthy one-to-one mapping.

This file remains only as provenance for the superseded freeze decision.
