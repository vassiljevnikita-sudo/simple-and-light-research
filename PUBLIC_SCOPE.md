# Public research scope

This repository is a **sanitized, history-free public snapshot** of the Dynamic-QBD /
Opportunity-Portfolio research program. The private repository `simple-and-light`
remains the internal source of truth. This mirror is a partial, curated export of it.

Source snapshot: `c196dba496203e803917fd4fff9a03d4df9dc3a2`
(branch `agent/dynamic-qbd-causal-model-store-v1`).

> **Status: partial mirror.** The sanitation contract below is satisfied, but the
> research tree has not been fully transferred. Several documents that the mirrored
> root docs link to are **not present in this repository yet**. See
> [Transfer status](#transfer-status) for the exact inventory before following a link.

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

> **Open breach.** The mirror's existing commits are authored and committed under
> the maintainer's private personal email address — the same address whose presence
> in commit metadata is the stated reason this repository does not import the private
> history. The published tree is clean; the commit metadata is not. This is tracked
> as incident `P9` in
> [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) and needs a maintainer decision,
> because correcting already-pushed commits means rewriting and force-pushing public
> history. Future commits should use a GitHub `users.noreply.github.com` address.

## Transfer status

Mirrored so far:

| Path | Kind |
|---|---|
| `README.md` | root documentation (sanitized) |
| `AGENTS.md` | agent instructions |
| `CURRENT_MODEL.md` | protected model identity |
| `DOCUMENTATION_AUTHORITY.md` | documentation precedence |
| `TOBECONTINUED.md` | work tracker (sanitized) |
| `QA_TOOL_FAILURE_LOG.md` | public-only sanitation ledger |
| `PUBLIC_SCOPE.md` | this contract |
| `stock_predictor/README.md` | package marker |
| `stock_predictor/__init__.py` | package initializer |
| `stock_predictor/backtests/opportunity_portfolio_research/contract_fingerprints.py` | research implementation |

**Not yet transferred.** The mirrored root documents reference the following paths.
They are accurate descriptions of the private source tree, but the targets do not
exist in this repository yet. Treat every link to them as pending, not broken
content:

| Referenced path | Referenced from |
|---|---|
| `research/DYNAMIC_QBD_NAMING.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED |
| `research/DYNAMIC_QBD_CURRENT_STATE.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED, CURRENT_MODEL |
| `research/DYNAMIC_QBD_RESULTS_INDEX.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED |
| `research/DYNAMIC_QBD_DATA_AND_RUNTIME.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED |
| `research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED |
| `research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md` | AGENTS, README, DOCUMENTATION_AUTHORITY, TOBECONTINUED |
| `research/DYNAMIC_QBD_SYSTEM_AUDIT_2026-08-15.md` | DOCUMENTATION_AUTHORITY |
| `stock_predictor/research_model_registry.json` | AGENTS, README, DOCUMENTATION_AUTHORITY, CURRENT_MODEL |
| `stock_predictor/project_manifest.json` | AGENTS, DOCUMENTATION_AUTHORITY |
| `stock_predictor/n25_dual_venue.py` | CURRENT_MODEL |
| `stock_predictor/benchmark_policy.py` | CURRENT_MODEL |
| `stock_predictor/backtests/opportunity_portfolio_research/README.md` | README |
| `stock_predictor/backtests/opportunity_portfolio_research/cost_contracts.py` | QA_TOOL_FAILURE_LOG |
| `docs/architecture/` | AGENTS, README |
| `artifacts/` and `artifacts/<run>/summary.json`, `REPORT.md`, `contract-audit.json`, `run-contract.json` | AGENTS, README, DOCUMENTATION_AUTHORITY |
| `agent.md` | DOCUMENTATION_AUTHORITY |

A reader who needs one of these documents must obtain it from the private
repository. An agent working in this mirror must not treat an absent document as
evidence that the corresponding contract does not exist.

## How remaining files should be transferred

Two transfer paths have been attempted from hosted agent sessions. Both are
constrained:

1. **Hosted write path.** Writing certain research source files into this repository
   through a hosted agent's file-creation or Git-data API is blocked by an upstream
   write-safety filter, independently of whether the file contains sensitive
   material. Neutral files in the same directory write successfully. Incidents
   `P3`-`P6` in [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) record the
   confirmed cases.
2. **Hosted read path.** A hosted session scoped to this public repository cannot
   read the private repository at all, so it cannot transfer files even when the
   write path would accept them. Incident `P7` records this.
3. **Hosted push path.** A hosted session may also have no write scope on this
   repository at all — both `git push` and the GitHub API return `403` while reads
   succeed. This is an authorization gap rather than a content filter, so it cannot
   be worked around by changing what is written. Incident `P10` records this.

The supported path for the remaining files is therefore **local and manual**:

1. check out the private repository at the source snapshot on a local machine;
2. create a new, history-free working directory for the public tree;
3. copy only files permitted by the inclusion contract, reviewing each against the
   exclusion contract;
4. re-run the sanitation probes in [Sanitation verification](#sanitation-verification);
5. commit and push directly to `simple-and-light-research` from that local machine.

Do not work around the write-safety filter by encoding, obfuscating, renaming or
altering executable logic. If a file cannot be transferred unchanged, transfer it
locally or leave it out and record why.
