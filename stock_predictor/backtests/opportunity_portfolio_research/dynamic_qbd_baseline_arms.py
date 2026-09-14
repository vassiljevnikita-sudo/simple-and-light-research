"""Selector-minimal causal baselines over the existing ModelStore boundary."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Callable

from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import ModelGenerationRecord
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor
from .dynamic_qbd_run_gate_contract import (FrozenOrchestratorPolicy, FrozenS3PoolRule,
                                             InitialSelectionRule, OrchestratorInputAsOf,
                                             GenerationOption)
from .contract_fingerprints import stable_hash


@dataclass(frozen=True)
class ArmState:
    decision_time: date
    active_generation_ids: tuple[str, ...]
    matched_portfolio_state_fingerprint: str
    active_generation_weights: tuple[tuple[str, float], ...] = ()
    pool_fingerprint: str = ""


class S0InitialStatic:
    """Initial causal generation selected at seed, then held fixed."""

    name = "S0_INITIAL_STATIC"

    def __init__(self, *, store: ModelStore, seed: date, generation_id: str) -> None:
        self.store = store
        self.seed = seed
        self.generation_id = generation_id
        if store.get_generation(generation_id).created_at > seed:
            raise ValueError("S0_INITIAL_GENERATION_AFTER_SEED")

    @classmethod
    def from_selection_artifact(cls, *, store: ModelStore, seed: date, horizon: int,
                                selection_rule: InitialSelectionRule,
                                selection_artifact: dict) -> "S0InitialStatic":
        recipe_id = selection_rule.validate_selection_artifact(selection_artifact, seed=seed)
        visible = store.generations_for_recipe_and_horizon(recipe_id, horizon, seed)
        if not visible:
            raise ValueError("S0_SELECTION_RECIPE_HAS_NO_SEED_GENERATION")
        return cls(store=store, seed=seed, generation_id=visible[-1].generation_id)

    def state_as_of(self, decision_time: date, portfolio_state_fingerprint: str) -> ArmState:
        visible = {record.generation_id for record in self.store.generations_as_of(decision_time)}
        if self.generation_id not in visible:
            raise ValueError("S0_INITIAL_GENERATION_NOT_VISIBLE")
        return ArmState(decision_time, (self.generation_id,), portfolio_state_fingerprint)


class S1AutoRefitSameRecipe:
    """Automatically use the newest visible generation of the fixed recipe."""

    name = "S1_AUTO_REFIT_SAME_RECIPE"

    def __init__(self, *, store: ModelStore, recipe_id: str, horizon: int) -> None:
        self.store = store
        self.recipe_id = recipe_id
        self.horizon = int(horizon)

    def state_as_of(self, decision_time: date, portfolio_state_fingerprint: str) -> ArmState:
        visible = self.store.generations_for_recipe_and_horizon(self.recipe_id, self.horizon, decision_time)
        if not visible:
            raise ValueError("S1_NO_VISIBLE_GENERATION")
        return ArmState(decision_time, (visible[-1].generation_id,), portfolio_state_fingerprint)


class S2aGenerationOnly:
    """Generation-only interface; selection is an explicit non-performance input."""

    name = "S2A_GENERATION_ONLY"

    def __init__(self, *, store: ModelStore, recipe_id: str, horizon: int) -> None:
        self.store = store
        self.recipe_id = recipe_id
        self.horizon = int(horizon)

    def visible_candidates(self, decision_time: date) -> tuple[ModelGenerationRecord, ...]:
        return self.store.generations_for_recipe_and_horizon(self.recipe_id, self.horizon, decision_time)

    def state_as_of(self, decision_time: date, portfolio_state_fingerprint: str,
                    selected_generation_id: str) -> ArmState:
        visible = {x.generation_id for x in self.visible_candidates(decision_time)}
        if selected_generation_id not in visible:
            raise ValueError("S2A_SELECTED_GENERATION_NOT_VISIBLE")
        return ArmState(decision_time, (selected_generation_id,), portfolio_state_fingerprint)


class S2bGenerationAndRecipe:
    """Recipe+generation interface; the caller supplies a predeclared selector."""

    name = "S2B_GENERATION_AND_RECIPE"

    def __init__(self, *, store: ModelStore, horizon: int | None = None) -> None:
        self.store = store
        self.horizon = None if horizon is None else int(horizon)

    def visible_candidates(self, decision_time: date) -> tuple[ModelGenerationRecord, ...]:
        visible = self.store.generations_as_of(decision_time)
        if self.horizon is not None:
            visible = tuple(record for record in visible if record.horizon == self.horizon)
        return visible

    def state_as_of(self, decision_time: date, portfolio_state_fingerprint: str,
                    selected_generation_id: str,
                    selector: Callable[[tuple[ModelGenerationRecord, ...]], str]) -> ArmState:
        candidates = self.visible_candidates(decision_time)
        selected = str(selector(candidates))
        if selected != selected_generation_id or selected not in {x.generation_id for x in candidates}:
            raise ValueError("S2B_SELECTOR_OUTPUT_NOT_VISIBLE_OR_NOT_REPRODUCIBLE")
        return ArmState(decision_time, (selected,), portfolio_state_fingerprint)

    def input_as_of(self, *, decision_time: date, incumbent_generation_id: str,
                    evidence: EvidenceCursor) -> OrchestratorInputAsOf:
        candidates = self.visible_candidates(decision_time)
        return OrchestratorInputAsOf(
            decision_time=decision_time,
            incumbent_generation_id=incumbent_generation_id,
            candidate_options=tuple(GenerationOption(x.generation_id, x.recipe_id, x.created_at)
                                     for x in candidates),
            evidence_fingerprint=evidence.fingerprint_as_of(decision_time),
        )

    def state_as_of_policy(self, *, inputs: OrchestratorInputAsOf,
                           portfolio_state_fingerprint: str,
                           policy: FrozenOrchestratorPolicy) -> ArmState:
        selected = policy.select(inputs)
        return self.state_as_of(inputs.decision_time, portfolio_state_fingerprint,
                                selected, lambda _: selected)


class S3EqualWeightPool:
    """Simple equal-weight pool over an explicit causal candidate set."""

    name = "S3_EQUAL_WEIGHT_POOL"

    def __init__(self, *, store: ModelStore, horizon: int | None = None) -> None:
        self.store = store
        self.horizon = None if horizon is None else int(horizon)

    def state_as_of(self, decision_time: date, portfolio_state_fingerprint: str,
                    candidate_generation_ids: Iterable[str]) -> ArmState:
        if self.horizon is None:
            raise ValueError("S3_HORIZON_REQUIRED")
        visible_records = self.store.generations_as_of(decision_time)
        if self.horizon is not None:
            visible_records = tuple(record for record in visible_records if record.horizon == self.horizon)
        visible = {x.generation_id for x in visible_records}
        selected = tuple(sorted({str(x) for x in candidate_generation_ids}))
        if not selected or not set(selected) <= visible:
            raise ValueError("S3_POOL_GENERATIONS_NOT_VISIBLE")
        weights = tuple((generation_id, 1.0 / len(selected)) for generation_id in selected)
        return ArmState(decision_time, selected, portfolio_state_fingerprint, weights,
                        stable_hash({"decision_time": decision_time, "generation_ids": selected,
                                     "weights": weights}))

    def state_as_of_rule(self, *, decision_time: date, portfolio_state_fingerprint: str,
                         rule: FrozenS3PoolRule) -> ArmState:
        if self.horizon is None:
            raise ValueError("S3_HORIZON_REQUIRED")
        if rule.candidate_scope != "CAUSALLY_VISIBLE_SAME_HORIZON":
            raise ValueError("S3_POOL_RULE_SCOPE_INVALID")
        visible = self.store.generations_as_of(decision_time)
        visible = tuple(record for record in visible if record.horizon == self.horizon)
        if not visible:
            raise ValueError("S3_POOL_NO_VISIBLE_GENERATION")
        if any(not record.source_generation_id or not record.source_generation_fingerprint or
               not record.source_generation_provenance for record in visible):
            raise ValueError("S3_POOL_INVALID_GENERATION_PROVENANCE")
        options = tuple(GenerationOption(x.generation_id, x.recipe_id, x.created_at)
                        for x in visible)
        selected = rule.select_pool(options)
        state = self.state_as_of(decision_time, portfolio_state_fingerprint, selected)
        return ArmState(state.decision_time, state.active_generation_ids,
                        state.matched_portfolio_state_fingerprint,
                        state.active_generation_weights,
                        stable_hash({"rule_hash": rule.rule_hash,
                                     "pool_fingerprint": state.pool_fingerprint}))


class OracleAvailability:
    """Expose the same available choices; realized outcomes remain outside causality."""

    name = "ORACLE_AVAILABILITY_ONLY"

    def __init__(self, *, store: ModelStore, horizon: int | None = None) -> None:
        self.store = store
        self.horizon = None if horizon is None else int(horizon)

    def available_generation_ids(self, decision_time: date) -> tuple[str, ...]:
        visible = self.store.generations_as_of(decision_time)
        if self.horizon is not None:
            visible = tuple(record for record in visible if record.horizon == self.horizon)
        return tuple(x.generation_id for x in visible)
