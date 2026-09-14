"""Frozen selection and evaluation gates required before a full pseudo-live run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract_fingerprints import stable_hash
from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END, HOLDOUT_BOUNDARY, PseudoLiveExperimentContract


_DEFAULT_EVALUATION_METRICS = (
    "terminal_wealth", "cagr", "benchmark_cagr", "cagr_excess",
    "relative_max_drawdown", "absolute_max_drawdown", "turnover", "trade_count",
    "transaction_costs", "expected_shortfall", "downside_deviation", "switch_count",
    "active_generation_age", "fraction_newest_generation_selected",
    "fraction_older_generation_retained", "recipe_concentration", "ticker_concentration",
    "oracle_regret", "captured_oracle_alpha",
)


def _date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _text(name: str, value: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name}_REQUIRED")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _freeze(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(_plain(value or {})))


@dataclass(frozen=True)
class InitialSelectionRule:
    rule_id: str = "INITIAL_SELECTION_RULE_V1"
    source: str = "candidate_oos.select_recipe_from_snapshot"
    selection_policy_version: str = "QBD_RECIPE_SELECTION_POLICY_V1"
    information_cutoff_field: str = "selection_cutoff"
    primary_metric: str = "FOLD_SPEARMAN_PREDICTION_VS_REALIZED_EXCESS"
    aggregation: str = "MEDIAN_FOLD_SPEARMAN"
    minimum_candidate_folds: int = 2
    tie_breakers: tuple[str, ...] = ("MEAN_FOLD_SPEARMAN", "POSITIVE_FOLD_FRACTION",
                                      "LOWER_MEAN_MAE", "CANDIDATE_ID")
    forbidden_inputs: tuple[str, ...] = ("CAGR", "SHARPE", "POST_SEED_PERFORMANCE", "FUTURE_RANK")

    def __post_init__(self) -> None:
        if self.source != "candidate_oos.select_recipe_from_snapshot":
            raise ValueError("INITIAL_SELECTION_SOURCE_NOT_CANONICAL")
        if self.information_cutoff_field != "selection_cutoff":
            raise ValueError("INITIAL_SELECTION_CUTOFF_FIELD_INVALID")
        if int(self.minimum_candidate_folds) < 2:
            raise ValueError("INITIAL_SELECTION_MINIMUM_FOLDS_TOO_LOW")

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            "rule_id": self.rule_id, "source": self.source,
            "selection_policy_version": self.selection_policy_version,
            "information_cutoff_field": self.information_cutoff_field,
            "primary_metric": self.primary_metric, "aggregation": self.aggregation,
            "minimum_candidate_folds": self.minimum_candidate_folds,
            "tie_breakers": self.tie_breakers, "forbidden_inputs": self.forbidden_inputs,
        })

    @property
    def rule_hash(self) -> str:
        return stable_hash(self.to_dict())

    def validate_selection_artifact(self, artifact: Mapping[str, Any], *, seed: date) -> str:
        if _date(artifact.get(self.information_cutoff_field, "")) != seed:
            raise ValueError("INITIAL_SELECTION_ARTIFACT_CUTOFF_MISMATCH")
        if str(artifact.get("selection_algorithm_version")) != self.selection_policy_version:
            raise ValueError("INITIAL_SELECTION_POLICY_VERSION_MISMATCH")
        if int(artifact.get("winner_evidence", {}).get("fold_count", 0)) < self.minimum_candidate_folds:
            raise ValueError("INITIAL_SELECTION_EVIDENCE_BELOW_MINIMUM")
        candidate_id = str(artifact.get("selected_candidate_id", ""))
        if not candidate_id:
            raise ValueError("INITIAL_SELECTION_CANDIDATE_MISSING")
        if any(key in artifact for key in self.forbidden_inputs):
            raise ValueError("INITIAL_SELECTION_FORBIDDEN_PERFORMANCE_INPUT")
        return candidate_id


@dataclass(frozen=True)
class GenerationOption:
    generation_id: str
    recipe_id: str
    created_at: date

    def __post_init__(self) -> None:
        _text("GENERATION_OPTION_ID", self.generation_id)
        _text("GENERATION_OPTION_RECIPE", self.recipe_id)
        object.__setattr__(self, "created_at", _date(self.created_at))
        if self.created_at > DEVELOPMENT_END:
            raise ValueError("GENERATION_OPTION_PROSPECTIVE_HOLDOUT_CLOSED")

    def to_dict(self) -> dict[str, Any]:
        return {"generation_id": self.generation_id, "recipe_id": self.recipe_id,
                "created_at": self.created_at.isoformat()}


@dataclass(frozen=True)
class OrchestratorInputAsOf:
    decision_time: date
    incumbent_generation_id: str
    candidate_options: tuple[GenerationOption, ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", _date(self.decision_time))
        if self.decision_time > DEVELOPMENT_END:
            raise ValueError("ORCHESTRATOR_INPUT_PROSPECTIVE_HOLDOUT_CLOSED")
        options = tuple(self.candidate_options)
        if not options or len({x.generation_id for x in options}) != len(options):
            raise ValueError("ORCHESTRATOR_INPUT_OPTIONS_INVALID")
        if any(x.created_at > self.decision_time for x in options):
            raise ValueError("ORCHESTRATOR_INPUT_CONTAINS_FUTURE_GENERATION")
        if self.incumbent_generation_id not in {x.generation_id for x in options}:
            raise ValueError("ORCHESTRATOR_INPUT_INCUMBENT_MISSING")
        _text("ORCHESTRATOR_INPUT_EVIDENCE_FINGERPRINT", self.evidence_fingerprint)
        object.__setattr__(self, "candidate_options", options)


@dataclass(frozen=True)
class FrozenOrchestratorPolicy:
    policy_id: str = "S2_ORCHESTRATOR_POLICY_V1"
    candidate_scope: str = "ALL_VISIBLE_GENERATIONS"
    scoring_formula: str = "NO_PERFORMANCE_SCORE"
    switch_rule: str = "ACTIVATE_NEWEST_VISIBLE_GENERATION"
    tie_break: tuple[str, ...] = ("CREATED_AT_ASC", "GENERATION_ID_ASC")
    allowed_input_schema: tuple[str, ...] = ("decision_time", "incumbent_generation_id",
                                               "candidate_options", "evidence_fingerprint")
    oracle_access: bool = False

    def __post_init__(self) -> None:
        if self.candidate_scope != "ALL_VISIBLE_GENERATIONS" or self.scoring_formula != "NO_PERFORMANCE_SCORE":
            raise ValueError("S2_POLICY_SCOPE_OR_SCORE_INVALID")
        if self.switch_rule != "ACTIVATE_NEWEST_VISIBLE_GENERATION":
            raise ValueError("S2_POLICY_SWITCH_RULE_INVALID")
        if self.oracle_access:
            raise ValueError("S2_POLICY_ORACLE_ACCESS_FORBIDDEN")

    def to_dict(self) -> dict[str, Any]:
        return _plain({"policy_id": self.policy_id, "candidate_scope": self.candidate_scope,
                       "scoring_formula": self.scoring_formula, "switch_rule": self.switch_rule,
                       "tie_break": self.tie_break, "allowed_input_schema": self.allowed_input_schema,
                       "oracle_access": self.oracle_access})

    @property
    def policy_hash(self) -> str:
        return stable_hash(self.to_dict())

    def select(self, inputs: OrchestratorInputAsOf) -> str:
        options = tuple(x for x in inputs.candidate_options
                        if x.generation_id != inputs.incumbent_generation_id)
        incumbent = next(x for x in inputs.candidate_options
                         if x.generation_id == inputs.incumbent_generation_id)
        newer = tuple(x for x in options if x.created_at > incumbent.created_at)
        if not newer:
            return incumbent.generation_id
        return sorted(newer, key=lambda x: (x.created_at, x.generation_id))[-1].generation_id


@dataclass(frozen=True)
class FrozenS3PoolRule:
    rule_id: str = "S3_POOL_RULE_V1"
    membership: str = "LATEST_VISIBLE_GENERATION_PER_RECIPE"
    weighting: str = "EQUAL_WEIGHT"
    candidate_scope: str = "CAUSALLY_VISIBLE_SAME_HORIZON"
    performance_based_membership: bool = False

    def __post_init__(self) -> None:
        if self.membership != "LATEST_VISIBLE_GENERATION_PER_RECIPE" or self.weighting != "EQUAL_WEIGHT":
            raise ValueError("S3_POOL_RULE_INVALID")
        if self.candidate_scope != "CAUSALLY_VISIBLE_SAME_HORIZON":
            raise ValueError("S3_POOL_RULE_SCOPE_INVALID")
        if self.performance_based_membership:
            raise ValueError("S3_POOL_PERFORMANCE_MEMBERSHIP_FORBIDDEN")

    def to_dict(self) -> dict[str, Any]:
        return _plain({"rule_id": self.rule_id, "membership": self.membership,
                       "weighting": self.weighting, "candidate_scope": self.candidate_scope,
                       "performance_based_membership": self.performance_based_membership})

    @property
    def rule_hash(self) -> str:
        return stable_hash(self.to_dict())

    def select_pool(self, options: tuple[GenerationOption, ...]) -> tuple[str, ...]:
        latest: dict[str, GenerationOption] = {}
        by_generation: dict[str, GenerationOption] = {}
        for option in options:
            prior_generation = by_generation.get(option.generation_id)
            if prior_generation is not None and (prior_generation.recipe_id != option.recipe_id or
                                                 prior_generation.created_at != option.created_at):
                raise ValueError("S3_POOL_DUPLICATE_GENERATION_ID_CONFLICT")
            by_generation[option.generation_id] = option
            prior = latest.get(option.recipe_id)
            if prior is None or (option.created_at, option.generation_id) > (prior.created_at, prior.generation_id):
                latest[option.recipe_id] = option
        return tuple(sorted(option.generation_id for option in latest.values()))


@dataclass(frozen=True)
class EvaluationContract:
    contract_id: str = "DQBD_CAUSAL_MODEL_STORE_EVALUATION_V1"
    primary_inference_unit: str = "PAIRED_PSEUDOLIVE_PATH_BY_SEED"
    bootstrap_method: str = "MOVING_CALENDAR_BLOCK_BOOTSTRAP"
    bootstrap_block_definition: str = "21_TRADING_SESSIONS"
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 212
    confidence_level: float = .95
    primary_scientific_question: str = "ORACLE_HEADROOM_EXISTS"
    primary_arm_contrast: str = "S2_MINUS_S0"
    secondary_arm_contrasts: tuple[str, ...] = ("S1_MINUS_S0", "S3_MINUS_S0", "S2_MINUS_S1")
    max_drawdown_guardrail: Mapping[str, Any] = None
    cost_treatment: str = "EXISTING_CANONICAL_COST_CONTRACT_IN_REPLAY"
    oracle_regret: str = "AVAILABLE_CHOICE_ORACLE_MINUS_CAUSAL_ARM"
    seed_robustness: tuple[str, ...] = ("SHORT", "PRIMARY", "LONG")
    concentration_diagnostics: tuple[str, ...] = ("TOP_1_TICKER_SHARE", "TOP_1_RECIPE_SHARE", "ACTIVE_GENERATION_COUNT")
    bootstrap_implementation: str = "dynamic_qbd_evaluation_bootstrap.moving_calendar_block_bootstrap"
    bootstrap_fingerprint: str = "53fb5ddcefb07013b008c40bd33a48c0045f32ef1aa70fc078cd4c8b8310a016"
    metric_schema: tuple[str, ...] = _DEFAULT_EVALUATION_METRICS
    metric_alias_mapping: Mapping[str, str] = None
    contrast_definitions: Mapping[str, Any] = None
    seed_definitions: Mapping[str, Any] = None
    oracle_definition: Mapping[str, Any] = None
    result_schema: tuple[str, ...] = (
        "summary.json", "REPORT.md", "run-contract.json", "contract-audit.json",
        "initial-model-store.csv/json", "generation-ledger.csv", "evidence-event-ledger.csv",
        "orchestrator-decisions.csv", "arm-comparison.csv", "event-subperiod-summary.csv",
        "oracle-regret-summary.csv", "seed-sensitivity.csv", "runtime-telemetry.json",
    )

    def __post_init__(self) -> None:
        if (int(self.bootstrap_repetitions) != 2000 or int(self.bootstrap_seed) != 212
                or self.bootstrap_block_definition != "21_TRADING_SESSIONS"
                or float(self.confidence_level) != 0.95):
            raise ValueError("EVALUATION_BOOTSTRAP_CONTRACT_INVALID")
        if set(self.seed_robustness) != {"SHORT", "PRIMARY", "LONG"}:
            raise ValueError("EVALUATION_SEED_ROBUSTNESS_INCOMPLETE")
        if self.bootstrap_implementation != "dynamic_qbd_evaluation_bootstrap.moving_calendar_block_bootstrap":
            raise ValueError("EVALUATION_BOOTSTRAP_IMPLEMENTATION_INVALID")
        if self.bootstrap_fingerprint == "TO_BE_RECOMPUTED":
            raise ValueError("EVALUATION_BOOTSTRAP_FINGERPRINT_REQUIRED")
        if len(self.bootstrap_fingerprint) != 64:
            raise ValueError("EVALUATION_BOOTSTRAP_FINGERPRINT_INVALID")
        if tuple(self.metric_schema) != _DEFAULT_EVALUATION_METRICS:
            raise ValueError("EVALUATION_METRIC_SCHEMA_INVALID")
        aliases = self.metric_alias_mapping or {
            "terminal_wealth": "terminal_value",
            "benchmark_cagr": "urth_cagr",
            "absolute_max_drawdown": "max_drawdown",
            "relative_max_drawdown": "worst_relative_drawdown",
            "transaction_costs": "total_cost_eur",
            "expected_shortfall": "expected_shortfall_95",
            "downside_deviation": "relative_downside_deviation",
        }
        expected_aliases = {
            "terminal_wealth": "terminal_value", "benchmark_cagr": "urth_cagr",
            "absolute_max_drawdown": "max_drawdown", "relative_max_drawdown": "worst_relative_drawdown",
            "transaction_costs": "total_cost_eur", "expected_shortfall": "expected_shortfall_95",
            "downside_deviation": "relative_downside_deviation",
        }
        if dict(aliases) != expected_aliases:
            raise ValueError("EVALUATION_METRIC_ALIAS_MAPPING_INVALID")
        object.__setattr__(self, "metric_alias_mapping", _freeze(aliases))
        defaults = {
            "S2_MINUS_S0": {"metric": "cagr_excess", "direction": "GREATER_THAN",
                            "estimator": "PAIRED_PATH_CAGR_EXCESS_CONTRAST",
                            "confidence_interval": "MOVING_CALENDAR_BLOCK_BOOTSTRAP_95_PERCENT",
                            "pass_fail": "PASS_IF_CI_LOWER_BOUND_GREATER_THAN_ZERO"},
            "S1_MINUS_S0": {"metric": "cagr_excess", "direction": "GREATER_THAN",
                            "estimator": "PAIRED_PATH_CAGR_EXCESS_CONTRAST",
                            "confidence_interval": "MOVING_CALENDAR_BLOCK_BOOTSTRAP_95_PERCENT",
                            "pass_fail": "REPORT_SECONDARY_CONTRAST_ONLY"},
            "S3_MINUS_S0": {"metric": "cagr_excess", "direction": "GREATER_THAN",
                            "estimator": "PAIRED_PATH_CAGR_EXCESS_CONTRAST",
                            "confidence_interval": "MOVING_CALENDAR_BLOCK_BOOTSTRAP_95_PERCENT",
                            "pass_fail": "REPORT_SECONDARY_CONTRAST_ONLY"},
            "S2_MINUS_S1": {"metric": "cagr_excess", "direction": "GREATER_THAN",
                            "estimator": "PAIRED_PATH_CAGR_EXCESS_CONTRAST",
                            "confidence_interval": "MOVING_CALENDAR_BLOCK_BOOTSTRAP_95_PERCENT",
                            "pass_fail": "REPORT_SECONDARY_CONTRAST_ONLY"},
        }
        contrasts = self.contrast_definitions or defaults
        if set(contrasts) != {self.primary_arm_contrast, *self.secondary_arm_contrasts}:
            raise ValueError("EVALUATION_CONTRAST_SCHEMA_INVALID")
        object.__setattr__(self, "contrast_definitions", _freeze(contrasts))
        seeds = self.seed_definitions or {"SHORT": "2020-08-31", "PRIMARY": "2021-02-26", "LONG": "2021-08-31"}
        if seeds != {"SHORT": "2020-08-31", "PRIMARY": "2021-02-26", "LONG": "2021-08-31"}:
            raise ValueError("EVALUATION_SEED_DEFINITIONS_INVALID")
        object.__setattr__(self, "seed_definitions", _freeze(seeds))
        oracle = self.oracle_definition or {
            "candidate_set": "EXACT_GENERATIONS_CAUSALLY_AVAILABLE_TO_S2_AT_T",
            "oracle_excess": "BEST_AVAILABLE_GENERATION_EXCESS_MINUS_URTH_EXCESS",
            "regret": "ORACLE_EXCESS_MINUS_CAUSAL_ARM_EXCESS",
            "captured_oracle_alpha": "CAUSAL_ARM_EXCESS / ORACLE_EXCESS_WHEN_ORACLE_EXCESS_GT_0",
            "role": "DIAGNOSTIC_ONLY_NO_SELECTION_INPUT_NO_EVENTS_NO_RECIPE_ACTIVATION",
        }
        if oracle.get("candidate_set") != "EXACT_GENERATIONS_CAUSALLY_AVAILABLE_TO_S2_AT_T":
            raise ValueError("EVALUATION_ORACLE_CANDIDATE_SET_INVALID")
        object.__setattr__(self, "oracle_definition", _freeze(oracle))
        object.__setattr__(self, "max_drawdown_guardrail", _freeze(self.max_drawdown_guardrail or {
            "metric": "max_drawdown", "limit": -.30,
            "comparison": "GREATER_OR_EQUAL", "interpretation": "REPORT_AND_GUARDRAIL_ONLY",
        }))

    def to_dict(self) -> dict[str, Any]:
        return _plain({"contract_id": self.contract_id, "primary_inference_unit": self.primary_inference_unit,
                       "bootstrap_method": self.bootstrap_method,
                       "bootstrap_block_definition": self.bootstrap_block_definition,
                       "bootstrap_repetitions": self.bootstrap_repetitions, "bootstrap_seed": self.bootstrap_seed,
                       "confidence_level": self.confidence_level,
                       "primary_scientific_question": self.primary_scientific_question,
                       "primary_arm_contrast": self.primary_arm_contrast,
                       "secondary_arm_contrasts": self.secondary_arm_contrasts,
                       "max_drawdown_guardrail": self.max_drawdown_guardrail,
                       "cost_treatment": self.cost_treatment, "oracle_regret": self.oracle_regret,
                       "seed_robustness": self.seed_robustness, "concentration_diagnostics": self.concentration_diagnostics,
                       "bootstrap_implementation": self.bootstrap_implementation,
                       "bootstrap_fingerprint": self.bootstrap_fingerprint,
                       "metric_schema": self.metric_schema, "metric_alias_mapping": self.metric_alias_mapping,
                       "contrast_definitions": self.contrast_definitions,
                       "seed_definitions": self.seed_definitions, "oracle_definition": self.oracle_definition,
                       "result_schema": self.result_schema})

    @property
    def contract_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class RunGateContract:
    base_contract_hash: str
    baseline_commit: str
    initial_selection: InitialSelectionRule
    orchestrator_policy: FrozenOrchestratorPolicy
    s3_pool_rule: FrozenS3PoolRule
    evaluation: EvaluationContract
    development_end: date = DEVELOPMENT_END
    holdout_boundary: date = HOLDOUT_BOUNDARY
    contract_id: str = "DQBD_CAUSAL_MODEL_STORE_V1_RUN_GATES"

    def __post_init__(self) -> None:
        if len(str(self.base_contract_hash)) != 64 or len(str(self.baseline_commit)) != 40:
            raise ValueError("RUN_GATE_BASELINE_OR_CONTRACT_HASH_INVALID")
        if _date(self.development_end) != DEVELOPMENT_END or _date(self.holdout_boundary) != HOLDOUT_BOUNDARY:
            raise ValueError("RUN_GATE_BOUNDARY_MISMATCH")
        object.__setattr__(self, "development_end", _date(self.development_end))
        object.__setattr__(self, "holdout_boundary", _date(self.holdout_boundary))

    def to_dict(self) -> dict[str, Any]:
        return _plain({"contract_id": self.contract_id, "base_contract_hash": self.base_contract_hash,
                       "baseline_commit": self.baseline_commit,
                       "development_end": self.development_end, "holdout_boundary": self.holdout_boundary,
                       "initial_selection": self.initial_selection.to_dict(),
                       "orchestrator_policy": self.orchestrator_policy.to_dict(),
                       "s3_pool_rule": self.s3_pool_rule.to_dict(),
                       "s3_pool_rule_hash": self.s3_pool_rule.rule_hash,
                       "evaluation": self.evaluation.to_dict()})

    @property
    def contract_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def decision_contract_hash(self) -> str:
        return self.contract_hash

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"contract_hash": self.contract_hash,
                                    "decision_contract_hash": self.decision_contract_hash}
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("contract_hash") != self.contract_hash:
                raise ValueError("DQBD_RUN_GATE_CONTRACT_IMMUTABLE")
            return self.contract_hash
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return self.contract_hash

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunGateContract":
        payload = dict(value)
        payload.pop("contract_hash", None)
        payload.pop("decision_contract_hash", None)
        s3_rule_hash = payload.pop("s3_pool_rule_hash", None)
        payload["initial_selection"] = InitialSelectionRule(**payload["initial_selection"])
        payload["orchestrator_policy"] = FrozenOrchestratorPolicy(**payload["orchestrator_policy"])
        payload["s3_pool_rule"] = FrozenS3PoolRule(**payload["s3_pool_rule"])
        if s3_rule_hash is not None and s3_rule_hash != payload["s3_pool_rule"].rule_hash:
            raise ValueError("DQBD_S3_POOL_RULE_HASH_MISMATCH")
        payload["evaluation"] = EvaluationContract(**payload["evaluation"])
        return cls(**payload)


def default_run_gate_contract(base_contract: PseudoLiveExperimentContract) -> RunGateContract:
    return RunGateContract(
        base_contract_hash=base_contract.contract_hash,
        baseline_commit=base_contract.baseline_commit,
        initial_selection=InitialSelectionRule(),
        orchestrator_policy=FrozenOrchestratorPolicy(),
        s3_pool_rule=FrozenS3PoolRule(),
        evaluation=EvaluationContract(),
    )
