# Public research scope

This repository is a **curated public snapshot** of the Dynamic-QBD /
Opportunity-Portfolio research program. The private repository `simple-and-light`
remains the internal source of truth. The published file tree is sanitized and the
private source history is not imported.

Source snapshot: `c196dba496203e803917fd4fff9a03d4df9dc3a2`
(branch `agent/dynamic-qbd-causal-model-store-v1`).

## Why the history is not mirrored

This repository starts from an empty history. The public copy is defined by the
inclusion contract below, not by the private repository: only files reviewed
against both contracts are copied in, and the private history is never imported
in any form.

Therefore:

- no `git filter-repo` / rewritten import of the private history;
- no merge, rebase, subtree or remote link to the private repository;
- every public commit is authored against reviewed content only.

## Exclusion contract

Anything that is not listed in the inclusion contract below is out of scope. In
particular, the following classes must **never** be published to this repository,
in tree content, commit messages, branch names or commit metadata.

### Credentials and operational configuration

- `.env` files and any environment-file fragment;
- any API key, token, session cookie or service credential;
- operational configuration that is not a research parameter.

### Third-party and business data

- any third-party, business or personal data;
- any automation, document or record that is not stock-selection research code,
  research documentation or compact research results.

### Personal and machine-local material

- private email addresses (the author's published contact address is not private;
  see `P9`);
- user-profile and home-directory paths and local user names;
- `.vs/` and other IDE/editor local state;
- machine-specific runtime configuration and operator scratch state.

### Private operational artifacts

- `output/` and other private run-output directories;
- private runtime, operations and incident artifacts belonging to the internal ledger;
- the private repository's historical QA / operational incident ledger
  (this mirror keeps its own public-only ledger instead);
- large local or licensed data stores (model, prediction, NAV and checkpoint stores).

### History

- the private repository's git history, in any rewritten or partial form.

## Inclusion contract

The following are in scope for this mirror:

- current Dynamic-QBD / Opportunity-Portfolio research documentation;
- research contracts, naming contracts and architecture plans;
- the relevant research implementation under
  `stock_predictor/backtests/opportunity_portfolio_research/` and its tests;
- selected **compact** result artifacts: summaries, contract audits and small
  evidence tables, never bulk prediction or checkpoint stores;
- root documentation that orients a reader or agent in the research program;
- research portfolio sizing and budget constraints, together with the named broker
  and market-data vendor. Sizing covers model-portfolio size, per-position target
  notional, capacity and rebalancing thresholds. These are research configuration
  parameters, explicitly approved for publication (see `P8` in
  [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md)); they are not private financial
  data and do not require redaction. Actual account balances, positions, holdings and
  order records remain excluded.

Anything transferred must be reviewed against the exclusion contract above at the
time of transfer. Membership in the inclusion contract does not by itself authorize
publication of a specific file.

## Sanitation verification

The published tree and history were last checked on 2026-10-30:

| Problem class | Probe | Scope | Result |
|---|---|---|---|
| Secrets | gitleaks | full git history | 0 findings |
| Private term list | case-insensitive search against a term list kept outside this repository | current tree | 0 matches |
| Personal email addresses | any address other than the author's published contact address and GitHub/Anthropic noreply addresses | current tree and commit metadata | 0 matches |
| User-profile paths and local user names | user-profile path, home-directory and user-name patterns | current tree | 0 matches |

Re-run these checks before every transfer batch, not only at the end. Scan commit
metadata as well as the tree: author and committer identity is part of the
published surface, and
`git log --all --format='%an <%ae> | %cn <%ce>' | sort -u` must contain only the
author's published contact address or GitHub/Anthropic noreply addresses.

Commit metadata (incident `P9` in [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md)):
Resolved 2026-10-30: the address in the commit metadata is the author's published
contact address (CV, GitHub profile, imprint of the Questify site); no history
rewrite needed. No other personal data may be committed.

## Transfer status

Transferred from the source snapshot, after per-file review against the exclusion
contract:

| Area | Contents |
|---|---|
| root docs | `README.md`, `AGENTS.md`, `agent.md`, `CURRENT_MODEL.md`, `DOCUMENTATION_AUTHORITY.md`, `TOBECONTINUED.md`, `PUBLIC_SCOPE.md`, `QA_TOOL_FAILURE_LOG.md` |
| `research/` | 18 files: naming contract, current state, results index, data/runtime, execution minimum baseline, causal Model-Store plan, audits and roadmaps, plus the two `DQBD_CAUSAL_MODEL_STORE_V1` JSON contracts |
| `docs/architecture/` | 5 implementation/runtime architecture documents |
| `docs/external-validation/` | 1 external validation document |
| `stock_predictor/` | `research_model_registry.json`, `project_manifest.json`, `n25_dual_venue.py`, `benchmark_policy.py`, package initializer and README |
| `stock_predictor/backtests/opportunity_portfolio_research/` | 241 files: Dynamic-QBD factory, replay, evidence, gates, schedulers, self-tests and package documentation |
| `artifacts/` | 64 compact evidence files: `summary.json`, `REPORT.md`, `contract-audit.json`, `run-contract.json` per run |

Verified on the published tree: all 209 Python files parse, all 37 JSON files are
valid, and every relative Markdown link resolves.

### Deliberately not transferred

These are excluded by contract, not pending. Do not treat their absence as an
oversight or transfer them later without a scope decision:

| Excluded | Reason |
|---|---|
| `artifacts/alpaca-minute/` (11 GB), `artifacts/massive-minute/` (2 GB) and the rest of the 14 GB artifact tree | large local/licensed data stores; only compact evidence is published |
| `docs/incidents/` | private-machine runtime incident history |
| operational configuration, credentials and IDE local state | never published |
| private operational notes, trackers and local logs | private operational artifacts |
| anything outside the stock-selection research program | not part of this research copy |
| the private git history | not part of the public snapshot |

An agent working in this mirror must not treat an absent document as evidence that
the corresponding contract does not exist. Read it in the private repository, and
never copy private material into this one unless it falls under the inclusion
contract and passes the review above.

## How transfers are performed

The transfer is done from a hosted session with read access to the private
repository and write access to this one: check out the source snapshot, select
files against the inclusion contract, review each against the exclusion contract,
run the sanitation probes, then commit and push.

Three constraints have been hit historically, all recorded in
[QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md):

1. **Content write filter.** Some research source files were blocked by an upstream
   write-safety filter when written through a hosted agent's file-creation or
   Git-data API, independently of their content (`P3`-`P6`). Pushing over plain git
   from a checkout is not subject to it.
2. **Read access.** A session scoped only to this public repository cannot read the
   private repository and therefore cannot transfer anything (`P7`).
3. **Write access.** A session's write scope on this repository can be absent for a
   time and then return with no change to the commit (`P10`). While it is absent,
   git and the GitHub API both return `403` and reads keep succeeding.

Constraints 2 and 3 are authorization states: re-verify them before concluding a
transfer is impossible. None of the three is ever worked around by encoding,
obfuscating, renaming or altering executable logic. If a file cannot be transferred
unchanged, transfer it over git from a checkout, or leave it out and record why.

Before each batch, re-run the probes in
[Sanitation verification](#sanitation-verification) on the files being added and
on commit metadata, not only at the end.
