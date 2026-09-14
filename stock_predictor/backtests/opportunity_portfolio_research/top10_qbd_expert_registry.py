"""Pinned R01-R10 universe from the verified frozen Top-10 manifest."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
from .top10_qbd_router_contracts import ExpertType, sha256_fingerprint

V4_CONTROLLER_ID = "TOP10_ADAPTATION_GATE_CONTROLLER_V4"
V4_CONTROLLER_SHA256 = "4c50231969c38f10a78ceec2f4d2e0619ff9d488c6dd1f80b78d4676874de6df"
_IDENTITIES = (
    ("R01", "R01_L_H11_D03_N1", "795e71385eba3bd0a67086aaa4151f624ace2a41df64834a31edad2ccfee3a36", 11, 3, 1, "6fce8a973eb49c7e", .28800938409656546, .975),
    ("R02", "R02_F_H24_D05_N1", "548fec6c3bbb328e6669a7432cab925fd07fcaf5ce9d4dd319492426db7d9350", 24, 5, 1, "6b26c82b02a626a4", .6667231283857229, .975),
    ("R03", "R03_L_H28_D21_N1", "0e2c7db987ce8a03ee1563a1fb53d8c84d4a725bec515d06cfc03b15b9126c23", 28, 21, 1, "8a72d08626b87757", .732763365673097, .975),
    ("R04", "R04_L_H28_D21_N5", "f49fdb8df360933ad9930f036517eb7b1d44d74b682b543422bc180be004dc9f", 28, 21, 5, "188334883af1a342", .732763365673097, .975),
    ("R05", "R05_L_H28_D21_N4", "81b4992595c33dd7d184ce20de3d47b241c2a1f89708975353a9ac87e3d2f8cf", 28, 21, 4, "451f4d7cc3c72c9b", .732763365673097, .975),
    ("R06", "R06_L_H28_D21_N6", "76e3e1c750852526b7c8d5bfb6b8a4b5459fc4e99461ba40c242c86d69f11d23", 28, 21, 6, "0a81416326322f6f", .732763365673097, .975),
    ("R07", "R07_L_H28_D21_N2", "f9de8531f157418147653597b124d03c06869e273d8de740e4cfa356f01c13b6", 28, 21, 2, "ce4f64a31103f970", .732763365673097, .975),
    ("R08", "R08_L_H28_D21_N3", "76273eec4271e3851ae04c9c9db70910e1c15d2b9fd2f2abb10b36d0ca748261", 28, 21, 3, "80cc9eda7189b2a4", .732763365673097, .975),
    ("R09", "R09_L_H24_D21_N5", "c6573b68ab420dd0bcaa37227719f2fb6d5162177d902bd3b1694cfb5d9544fc", 24, 21, 5, "e357ea67755c1611", .5800546365818118, .95),
    ("R10", "R10_L_H24_D21_N6", "a8e7aa302f7cf90e4832bd9ffb676de0d6a689a597ad769abedbffadc4b46b4e", 24, 21, 6, "48648db76dddcd28", .5800546365818118, .95),
)

@dataclass(frozen=True)
class FrozenEntryPolicy:
    policy_id: str
    policy_hash: str
    resolved_threshold: float
    score_quantile: float
    top_fraction: float
    max_names: int

@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    expert_type: ExpertType
    model_artifact_hash: str | None = None
    horizon: int | None = None
    holding_days: int | None = None
    max_names: int | None = None
    entry_policy_id: str | None = None
    exit_policy_id: str | None = None
    feature_schema_hash: str | None = None
    adaptation_controller_id: str | None = None
    adaptation_controller_hash: str | None = None
    entry_policy: FrozenEntryPolicy | None = None

def build_top10_qbd_registry() -> tuple[ExpertSpec, ...]:
    return tuple(ExpertSpec(
        expert_id=eid, expert_type=ExpertType.STOCK, model_artifact_hash=artifact,
        horizon=horizon, holding_days=holding, max_names=max_names,
        entry_policy_id=f"TOP10_FROZEN_ENTRY_{policy_id}",
        exit_policy_id="LEARNED_EXIT" if "_L_" in model_id else "FIXED_HORIZON",
        feature_schema_hash="6ad49e90ed1120c17140c240c29b1914a34bf2cf3d55539e08471b4016a5c0e6",
        adaptation_controller_id=V4_CONTROLLER_ID, adaptation_controller_hash=V4_CONTROLLER_SHA256,
        entry_policy=FrozenEntryPolicy(policy_id, sha256_fingerprint((policy_id, threshold, quantile, .005, max_names)), threshold, quantile, .005, max_names))
        for eid, model_id, artifact, horizon, holding, max_names, policy_id, threshold, quantile in _IDENTITIES) + (
        ExpertSpec("MSCI_WORLD", ExpertType.BENCHMARK, entry_policy_id="BENCHMARK_HOLD"),
        ExpertSpec("CASH", ExpertType.CASH, entry_policy_id="CASH_HOLD"),)

def validate_registry(registry: Sequence[ExpertSpec]) -> None:
    ids = [x.expert_id for x in registry]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate expert IDs")
    required = {"R%02d" % i for i in range(1, 11)} | {"MSCI_WORLD", "CASH"}
    if set(ids) != required:
        raise ValueError(f"registry must contain exact R01-R10 plus fallbacks: {sorted(set(ids))}")
    for spec in registry:
        if spec.expert_type == ExpertType.STOCK and any(not getattr(spec, f) for f in (
            "model_artifact_hash", "feature_schema_hash", "adaptation_controller_id", "adaptation_controller_hash")):
            raise ValueError(f"missing stock identity for {spec.expert_id}")
        if spec.expert_type == ExpertType.STOCK and any(str(getattr(spec, f)).upper().startswith("UNRESOLVED") for f in (
            "model_artifact_hash", "feature_schema_hash", "adaptation_controller_hash")):
            raise ValueError(f"unresolved stock identity for {spec.expert_id}")

def registry_hash(registry: Sequence[ExpertSpec]) -> str:
    validate_registry(registry)
    return sha256_fingerprint(tuple(asdict(x) for x in registry))
