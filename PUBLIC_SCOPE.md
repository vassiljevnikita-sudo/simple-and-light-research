# Public research scope

This repository is a **curated public snapshot** of the Dynamic-QBD /
Opportunity-Portfolio research program. The private repository `simple-and-light`
remains the internal source of truth. The published file tree is sanitized and the
private source history is not imported. Public-mirror commit-metadata sanitation is
a separate publication invariant tracked in `QA_TOOL_FAILURE_LOG.md` / `TOBECONTINUED.md`.

Source snapshot: `c196dba496203e803917fd4fff9a03d4df9dc3a2`
(branch `agent/dynamic-qbd-causal-model-store-v1`).

> **Status: research scope transferred; commit-metadata sanitation still open.**
> Every document in the documented public research scope is present, and the file
> tree passes the defined sanitation probes. The remaining publication blocker is
> `P9`: early public-mirror commits expose a private author/committer address. The
> mirror must not be declared fully sanitized until the reachable public history is
> rebuilt/replaced and the metadata probe passes.

## Why the history is not mirrored

This repository intentionally starts from an empty history. The private repository's
commit history is **not** importable, because commit metadata and historical tree
contents contain credentials, customer data and a private personal email address.
Sanitizing the published tree is not sufficient when the history carries the same
material.

Therefore:

- no `git filter-repo` / rewritten import of the private history;
- no merge, rebase, subtree or remote link to the private repository;
- every public commit is authored against sanitized content only.

## Exclusion contract

The following classes must **never** be published to this repository, in tree
content, commit messages, branch names or commit metadata.

### Credentials and configuration secrets

- `.env` files and any environment-file fragment;
- Notion integration tokens and workspace identifiers;
- Shopware credentials, including `SHOPWARE_CLIENT_SECRET` and any client ID/secret pair;
- any API key, bearer token, session cookie or broker credential.

### Customer and business data

- customer DOCX documents and generated customer correspondence;
- customer numbers, order numbers (`Bestellnummern`) and order/customer logs;
- Herbery business automation and its data;
- dunning / `Mahnungen` automation, templates and records;
- Shopware order-processing automation.

### Personal and machine-local material

- personal email addresses, including the maintainer's private Gmail address;
- local Windows and OneDrive paths (`C:\Users\...` and equivalents);
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
- selected **compact** result artifacts — summaries, contract audits and small
  evidence tables, never bulk prediction or checkpoint stores;
- root documentation that orients a reader or agent in the research program;
- research portfolio sizing and budget constraints — model-portfolio size,
  per-position target notional, capacity and rebalancing thresholds — together with
  the named broker and market-data vendor. These are research configuration
  parameters, explicitly approved for publication (see `P8` in
  [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md)); they are not private financial
  data and do not require redaction. Actual account balances, positions, holdings and
  order records remain excluded.

Anything transferred must be reviewed against the exclusion contract above at the
time of transfer. Membership in the inclusion contract does not by itself authorize
publication of a specific file.

## Sanitation verification

The published tree has been checked against the known problem classes. At the time
of writing there are no matches for:

| Problem class | Probe | Result |
|---|---|---|
| Notion tokens | integration-token patterns | no match |
| Shopware secrets | `SHOPWARE_CLIENT_SECRET` | no match |
| Private email | maintainer's private Gmail address | no match |
| Local machine paths | `C:\Users\...` | no match |
| Internal tooling marker | `plotn` | no match |
| Business automation | `Herbery` | no match |
| Dunning automation | `Mahnungen` | no match |
| Customer/order data | order and customer number patterns | no match |

Re-run this check before every transfer batch, not only at the end.

Two things about running it:

- **Exclude this file from the probes.** `PUBLIC_SCOPE.md` names every problem class
  in order to define the exclusion contract, so an unfiltered scan will always match
  against it and hide real hits. Scan with this document excluded.
- **Scan commit metadata, not only the tree.** Author and committer identity is part
  of the published surface. `git log --all --format='%an <%ae> | %cn <%ce>' | sort -u`
  must not contain a private address.

> **Known publication blocker.** Early public-mirror commits contain the
> maintainer's private personal email address in author/committer metadata. The
> published file tree is clean; the reachable public commit graph is not. The earlier
> decision to accept this was superseded by the maintainer on 2026-09-14. Tracked as
> incident `P9` in [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md). Required
> closure: rebuild or recreate the public history with non-private commit metadata,
> remove/update every public ref that still reaches the affected commits, and rerun
> the metadata probe before marking sanitation complete.

## Transfer status

Transferred from the source snapshot, after per-file review against the exclusion
contract:

| Area | Contents |
|---|---|
| root docs | `README.md`, `AGENTS.md`, `agent.md`, `CURRENT_MODEL.md`, `DOCUMENTATION_AUTHORITY.md`, `TOBECONTINUED.md`, `PUBLIC_SCOPE.md`, `QA_TOOL_FAILURE_LOG.md` |
| `research/` | 18 files — naming contract, current state, results index, data/runtime, execution minimum baseline, causal Model-Store plan, audits and roadmaps, plus the two `DQBD_CAUSAL_MODEL_STORE_V1` JSON contracts |
| `docs/architecture/` | 5 implementation/runtime architecture documents |
| `docs/external-validation/` | 1 external validation document |
| `stock_predictor/` | `research_model_registry.json`, `project_manifest.json`, `n25_dual_venue.py`, `benchmark_policy.py`, package initializer and README |
| `stock_predictor/backtests/opportunity_portfolio_research/` | 241 files — Dynamic-QBD factory, replay, evidence, gates, schedulers, self-tests and package documentation |
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
| `ops/`, `output/`, `scripts/`, `Misc/`, `Old/`, `Images/` | private operational and scratch material |
| `.env`, `.env.example`, `.vs/` | secrets and IDE local state |
| Herbery, Shopware, Notion and dunning automation at the repository root | customer and business data |
| `CLAUDE.md`, repair/status trackers, local logs | private operational artifacts |
| the private git history | commit metadata carries credentials, customer data and a private address |

An agent working in this mirror must not treat an absent document as evidence that
the corresponding contract does not exist — read it in the private repository.

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
   time — `403` on both git and the GitHub API while reads keep succeeding — and
   then return with no change to the commit (`P10`).

Constraints 2 and 3 are authorization states: re-verify them before concluding a
transfer is impossible. None of the three is ever worked around by encoding,
obfuscating, renaming or altering executable logic. If a file cannot be transferred
unchanged, transfer it over git from a checkout, or leave it out and record why.

Before each batch, re-run the probes in
[Sanitation verification](#sanitation-verification) — on the files being added, and
on commit metadata, not only at the end.
