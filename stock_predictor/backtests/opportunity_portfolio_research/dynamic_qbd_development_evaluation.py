"""Development runner for factory A/B/C, family evidence and Gate 1."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .contract_fingerprints import stable_hash
from .dynamic_qbd_abc_schedules import compare_abc, validate_abc_generation_schedules
from .dynamic_qbd_generation_contracts import ExpertFamilySpec, FactoryArm, PRIMARY_FACTORY_ARMS
from .dynamic_qbd_evidence import build_monthly_family_evidence
from .dynamic_qbd_gate1 import run_gate1
from .dynamic_qbd_incremental_gates import run_gate1b, run_gate2, run_gate3
from .dynamic_qbd_portfolio_replay import generation_authoritative_signals, replay_family
from .dynamic_qbd_wealth_metrics import wealth_path_metrics


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publish_structured_artifact(source: Path, directory: Path) -> None:
    """Expose one canonical result in the documented local-run directory tree."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def tax_config_from_contract(contract: dict) -> TaxConfig:
    mode=str(contract.get("mode", "PRE_TAX")).upper()
    enabled=bool(contract.get("enabled", False))
    if mode == "PRE_TAX":
        if enabled:
            raise ValueError("PRE_TAX_CONTRACT_CANNOT_BE_ENABLED")
        return TaxConfig(enabled=False)
    if mode == "POST_TAX":
        if not enabled or str(contract.get("engine", "DE_RETAIL_APPROX")).upper() != "DE_RETAIL_APPROX":
            raise ValueError("POST_TAX_CONTRACT_REQUIRES_DE_RETAIL_APPROX_ENGINE")
        mode = "DE_RETAIL_APPROX"
    if mode != "DE_RETAIL_APPROX" or not enabled:
        raise ValueError("TAX_CONTRACT_MUST_BE_PRE_TAX_OR_ENABLED_DE_RETAIL_APPROX")
    return TaxConfig(
        enabled=True,
        capital_gains_rate=float(contract.get("capital_gains_rate", .25)),
        solidarity_surcharge=float(contract.get("solidarity_surcharge", .055)),
        allowance_eur=float(contract.get("allowance_eur", 1000.0)),
        church_tax_rate=float(contract.get("church_tax_rate", 0.0)),
    )


class _StreamingParquetWriter:
    """Append bounded pandas frames and publish one Parquet file atomically."""

    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self.temporary = self.destination.with_name(self.destination.name + f".{os.getpid()}.tmp")
        self.writer = None
        self.rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.temporary, table.schema, compression="zstd")
        elif table.schema != self.writer.schema:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += int(table.num_rows)

    def publish(self) -> int:
        try:
            if self.writer is None:
                pd.DataFrame().to_parquet(self.temporary, index=False)
            else:
                self.writer.close()
                self.writer = None
            os.replace(self.temporary, self.destination)
            return self.rows
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            if self.temporary.exists():
                self.temporary.unlink()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _family_replay_cache_key(family, schedule: pd.DataFrame) -> str:
    return stable_hash({
        "schema_version": "DYNAMIC_QBD_REPLAY_FAMILY_CACHE_V1",
        "family": asdict(family),
        "schedule": json.loads(schedule.to_json(orient="records", date_format="iso")),
    })


def _family_replay_cache_paths(root: Path, family_id: str) -> dict[str, Path]:
    cache_root = root / "cache" / "development_replay"
    safe = str(family_id).replace("/", "_").replace("\\", "_")
    return {
        "nav": cache_root / f"{safe}.nav.parquet",
        "trades": cache_root / f"{safe}.trades.parquet",
        "positions": cache_root / f"{safe}.positions.parquet",
        "monthly": cache_root / f"{safe}.monthly.parquet",
        "meta": cache_root / f"{safe}.json",
    }


def _load_family_replay_cache(root: Path, family_id: str, cache_key: str) -> dict | None:
    paths = _family_replay_cache_paths(root, family_id)
    if not paths["meta"].is_file():
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if meta.get("cache_key") != cache_key or meta.get("family_id") != str(family_id):
            return None
        for name in ("nav", "trades", "positions", "monthly"):
            if not paths[name].is_file():
                return None
            if int(pq.ParquetFile(paths[name]).metadata.num_rows) != int(meta.get(f"{name}_rows", -1)):
                return None
        return {
            "nav": pd.read_parquet(paths["nav"]),
            "trades": pd.read_parquet(paths["trades"]),
            "positions": pd.read_parquet(paths["positions"]),
            "monthly": pd.read_parquet(paths["monthly"]),
            "summaries": meta.get("summaries", []),
            "replay_states": meta.get("replay_states", []),
            "max_date": meta.get("max_date"),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _publish_family_replay_cache(root: Path, family_id: str, cache_key: str, *, nav: pd.DataFrame,
                                 trades: pd.DataFrame, positions: pd.DataFrame, monthly: pd.DataFrame,
                                 summaries: list[dict], replay_states: list[dict], max_date) -> dict:
    paths = _family_replay_cache_paths(root, family_id)
    for name, frame in (("nav", nav), ("trades", trades), ("positions", positions), ("monthly", monthly)):
        temporary = paths[name].with_name(paths[name].name + f".{os.getpid()}.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, paths[name])
    meta = {
        "schema_version": "DYNAMIC_QBD_REPLAY_FAMILY_CACHE_V1",
        "family_id": str(family_id), "cache_key": cache_key,
        "nav_rows": len(nav), "trades_rows": len(trades),
        "positions_rows": len(positions), "monthly_rows": len(monthly),
        "summaries": summaries, "replay_states": replay_states,
        "max_date": str(max_date),
    }
    _atomic_json(paths["meta"], meta)
    return {"nav": nav, "trades": trades, "positions": positions, "monthly": monthly,
            "summaries": summaries, "replay_states": replay_states, "max_date": str(max_date)}


def run_development(*, families, predictions, prices, generation_schedule, output_root, matured_predictions=None, initial=10000.0,
                    final_holdout_start=date(2026,7,25), market_state=None, holdout_contract=None):
    if holdout_contract not in {"PRESERVE_HISTORICAL_LOCKBOX", "PROSPECTIVE_FROM_2026_07_25"}:
        raise ValueError("EXPLICIT_DYNAMIC_QBD_HOLDOUT_CONTRACT_REQUIRED")
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    holdout=pd.Timestamp(final_holdout_start)
    prediction_dates = (predictions.max_date() if hasattr(predictions, "max_date") else
                        pd.to_datetime(predictions["decision_date"]).max())
    if pd.Timestamp(prediction_dates) >= holdout:
        raise PermissionError(f"DYNAMIC_QBD_FINAL_HOLDOUT_LOCKED:PREDICTIONS:{final_holdout_start}")
    checks=((prices,"date","PRICES"),(generation_schedule,"activation_date","GENERATIONS"))
    for frame,column,label in checks:
        if column not in frame or pd.to_datetime(frame[column]).ge(holdout).any():
            raise PermissionError(f"DYNAMIC_QBD_FINAL_HOLDOUT_LOCKED:{label}:{final_holdout_start}")
    if market_state is not None and ("assessment_date" not in market_state or
                                     pd.to_datetime(market_state["assessment_date"]).ge(holdout).any()):
        raise PermissionError(f"DYNAMIC_QBD_FINAL_HOLDOUT_LOCKED:MARKET_STATE:{final_holdout_start}")
    if matured_predictions is not None:
        matured_max = (matured_predictions.max_terminal_date() if hasattr(matured_predictions, "max_terminal_date") else
                       pd.to_datetime(matured_predictions["terminal_date"]).max() if "terminal_date" in matured_predictions else holdout)
        if pd.Timestamp(matured_max) >= holdout:
            raise PermissionError(f"DYNAMIC_QBD_FINAL_HOLDOUT_LOCKED:MATURED_PREDICTIONS:{final_holdout_start}")
    if "lifecycle_status" in generation_schedule and not generation_schedule["lifecycle_status"].astype(str).isin(("VALID","GenerationStatus.VALID")).all():
        raise PermissionError("NON_VALID_GENERATION_HAS_REPLAY_AUTHORITY")
    validate_abc_generation_schedules(generation_schedule)
    family_payload={"schema_version":"DYNAMIC_QBD_FAMILY_REGISTRY_V1","families":[asdict(x) for x in families]}
    family_payload["family_registry_hash"]=stable_hash(family_payload)
    _write_json(output_root/"family_registry.json", family_payload)
    generation_schedule.to_parquet(output_root/"generation_registry.parquet",index=False)
    generation_schedule.to_parquet(output_root/"refit_history.parquet",index=False)
    calibration_columns=[x for x in ("family_id","generation_id","activation_date","resolved_threshold","calibration_fingerprint","information_cutoff") if x in generation_schedule]
    generation_schedule[calibration_columns].to_parquet(output_root/"calibration_history.parquet",index=False)
    _write_json(output_root/"generation_fingerprints.json", {str(row.generation_id):str(getattr(row,"generation_fingerprint", "")) for row in generation_schedule.itertuples()})
    nav_writer = _StreamingParquetWriter(output_root / "family_shadow_nav.parquet")
    trade_writer = _StreamingParquetWriter(output_root / "family_shadow_trades.parquet")
    position_writer = _StreamingParquetWriter(output_root / "family_shadow_positions.parquet")
    relative_writer = _StreamingParquetWriter(output_root / "family_relative_wealth.parquet")
    monthly_writer = _StreamingParquetWriter(output_root / "abc_monthly_evidence.parquet")
    summaries, replay_states = [], []
    max_nav_date = None
    prediction_cache_key = None
    prediction_cache = None
    authoritative_signal_cache = {}
    for family in families:
        fs = generation_schedule.loc[generation_schedule["family_id"].eq(family.family_id)]
        family_cache_key = _family_replay_cache_key(family, fs)
        cached = _load_family_replay_cache(output_root, family.family_id, family_cache_key)
        if cached is not None:
            nav_writer.append(cached["nav"])
            trade_writer.append(cached["trades"])
            position_writer.append(cached["positions"])
            relative_writer.append(cached["nav"].assign(
                relative_wealth=cached["nav"]["strategy_value"] / cached["nav"]["urth_value"]))
            monthly_writer.append(cached["monthly"])
            summaries.extend(cached["summaries"])
            replay_states.extend(cached["replay_states"])
            if cached.get("max_date"):
                max_nav_date = max(max_nav_date, str(cached["max_date"])) if max_nav_date else str(cached["max_date"])
            continue
        prediction_columns = ["decision_date", "ticker", "score", "model_artifact_id"]
        missing_prediction_columns = set(prediction_columns) - set(predictions)
        if missing_prediction_columns:
            raise ValueError(f"GENERATION_PREDICTIONS_MISSING:{sorted(missing_prediction_columns)}")
        model_ids = set(fs["model_artifact_id"].astype(str))
        key = tuple(sorted(model_ids))
        if key != prediction_cache_key:
            prediction_cache = (predictions.read_model_ids(key) if hasattr(predictions, "read_model_ids") else
                                predictions.loc[predictions["model_artifact_id"].astype(str).isin(model_ids), prediction_columns])
            prediction_cache_key = key
            authoritative_signal_cache = {}
        family_predictions = prediction_cache
        family_nav, family_trades, family_positions = [], [], []
        family_summaries, family_replay_states = [], []
        available_arms = set(fs["arm"].astype(str))
        replay_arms = list(PRIMARY_FACTORY_ARMS) + [
            arm for arm in (
                FactoryArm.B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION,
                FactoryArm.C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION,
            ) if arm.value in available_arms
        ]
        for arm in replay_arms:
            schedule = fs.loc[fs["arm"].eq(arm.value)].copy()
            if schedule.empty:
                raise ValueError(f"MISSING_ABC_SCHEDULE:{family.family_id}:{arm.value}")
            entry = family.entry_policy_rule
            quantiles = tuple(entry.get("score_quantiles", (entry.get("score_quantile", family.threshold_rule.get("score_quantile",.975)),)))
            fractions = tuple(entry.get("top_fractions", (entry.get("top_fraction",.005),)))
            policy = Policy(family.horizon_sessions, float(quantiles[0]),
                            float(fractions[0]), family.max_names, family.holding_days,
                            str(family.exit_policy.get("family","FIXED")), float(family.exit_policy.get("value",0.0)),
                            str(family.exit_policy.get("replacement","IGNORE_NEW")), str(entry.get("allocation","EQUAL_ACTIVE")),
                            float(entry.get("sleeve",.50)))
            authority_key = tuple(
                (str(row.activation_date), str(row.model_artifact_id))
                for row in schedule[["activation_date", "model_artifact_id"]].sort_values("activation_date").itertuples(index=False)
            )
            authoritative = authoritative_signal_cache.get(authority_key)
            if authoritative is None:
                authoritative = generation_authoritative_signals(family_predictions, schedule)
                authoritative_signal_cache[authority_key] = authoritative
            try:
                result = replay_family(signals=family_predictions, prices=prices, policy=policy, generation_schedule=schedule,
                                       cost=CostModel(float(family.cost_contract.get("roundtrip_bps",20))),
                                       tax=tax_config_from_contract(family.tax_contract), initial=initial,
                                       authoritative_signals=authoritative)
            except Exception as exc:
                raise type(exc)(f"{family.family_id}:{arm.value}:{exc}").with_traceback(exc.__traceback__) from exc
            operational_state=dict(result.get("replay_state",{}))
            operational_state.pop("curve",None); operational_state.pop("trades",None)
            family_replay_states.append({"family_id":family.family_id,"arm":arm.value,"state":operational_state})
            curve = result["curve"].copy(); curve["family_id"] = family.family_id; curve["arm"] = arm.value; family_nav.append(curve)
            trade = pd.DataFrame(result.get("trades", []));
            if not trade.empty: trade["family_id"] = family.family_id; trade["arm"] = arm.value; family_trades.append(trade)
            position = pd.DataFrame(result.get("open_positions", []));
            if not position.empty: position["family_id"] = family.family_id; position["arm"] = arm.value; family_positions.append(position)
            risk = wealth_path_metrics(curve)
            family_summaries.append({"family_id":family.family_id,"arm":arm.value,"terminal_value":result["metrics"]["terminal_value"],
                              "benchmark_terminal_value":result["metrics"]["urth_terminal_value"],"trade_count":result["metrics"]["trade_count"],
                              "turnover":result["metrics"]["turnover"],"costs":result["metrics"]["total_cost_eur"],
                              "max_positions":result["metrics"].get("max_positions",0),
                              "max_stock_exposure":float(curve["stock_exposure"].max()) if len(curve) else 0.0,
                              "sleeve_mark_to_market_breach_days":result["metrics"].get("sleeve_mark_to_market_breach_days",0),
                              "sleeve_contract":entry.get("sleeve_contract"),
                              "tax_world":result["metrics"].get("tax_world"),**risk})
        family_nav_frame = pd.concat(family_nav, ignore_index=True)
        family_trade_frame = pd.concat(family_trades, ignore_index=True) if family_trades else pd.DataFrame()
        family_position_frame = pd.concat(family_positions, ignore_index=True) if family_positions else pd.DataFrame()
        family_monthly = family_nav_frame.copy().sort_values(["family_id", "arm", "date"])
        family_monthly["assessment_date"] = pd.to_datetime(family_monthly["date"]).dt.to_period("M").dt.to_timestamp("M")
        family_monthly = family_monthly.groupby(["family_id", "arm", "assessment_date"], as_index=False).tail(1)
        family_monthly["relative_wealth"] = family_monthly["strategy_value"] / family_monthly["urth_value"]
        family_monthly["relative_return"] = family_monthly.groupby(["family_id", "arm"])["relative_wealth"].pct_change()
        family_monthly = family_monthly.dropna(subset=["relative_return"])
        cached = _publish_family_replay_cache(
            output_root, family.family_id, family_cache_key, nav=family_nav_frame,
            trades=family_trade_frame, positions=family_position_frame, monthly=family_monthly,
            summaries=family_summaries, replay_states=family_replay_states,
            max_date=pd.to_datetime(family_nav_frame["date"]).max())
        nav_writer.append(family_nav_frame)
        trade_writer.append(family_trade_frame)
        position_writer.append(family_position_frame)
        relative_writer.append(family_nav_frame.assign(
            relative_wealth=family_nav_frame["strategy_value"] / family_nav_frame["urth_value"]))
        monthly_writer.append(family_monthly)
        summaries.extend(family_summaries)
        replay_states.extend(family_replay_states)
        family_max_date = str(pd.to_datetime(family_nav_frame["date"]).max())
        max_nav_date = max(max_nav_date, family_max_date) if max_nav_date else family_max_date
    nav_writer.publish(); trade_writer.publish(); position_writer.publish(); relative_writer.publish(); monthly_writer.publish()
    prediction_cache = None
    summary_frame = pd.DataFrame(summaries)
    nav_frame = None
    trade_frame = None
    position_frame = None
    if matured_predictions is not None:
        destination = output_root/"family_matured_predictions.parquet"
        if hasattr(matured_predictions, "copy_to_parquet"):
            matured_predictions.copy_to_parquet(destination)
        else:
            matured_predictions.to_parquet(destination,index=False)
    monthly = pd.read_parquet(output_root / "abc_monthly_evidence.parquet")
    summary_frame.to_csv(output_root/"abc_arm_summary.csv",index=False); abc=compare_abc(monthly); _write_json(output_root/"abc_decision.json",abc)
    summary_frame[[x for x in ("family_id","arm","relative_max_drawdown","cdar_95","drawdown_duration_sessions","median_recovery_duration_sessions","time_under_water_fraction","capital_impairment_area") if x in summary_frame]].to_csv(output_root/"family_drawdown_metrics.csv",index=False)
    summary_frame[[x for x in ("family_id","arm","relative_sortino","relative_downside_deviation","expected_shortfall_95") if x in summary_frame]].to_csv(output_root/"family_downside_metrics.csv",index=False)
    epistemic_columns=[x for x in ("family_id","arm","trade_count","top_trade_contribution","top_ticker_contribution","fold_stability","generation_stability") if x in summary_frame]
    summary_frame[epistemic_columns].to_csv(output_root/"family_epistemic_diagnostics.csv",index=False)
    matured_outcomes = output_root / "family_matured_outcomes.parquet"
    if matured_outcomes.exists():
        matured_outcomes.unlink()
    try:
        os.link(output_root / "family_shadow_trades.parquet", matured_outcomes)
    except OSError:
        shutil.copy2(output_root / "family_shadow_trades.parquet", matured_outcomes)
    c_nav=pd.read_parquet(output_root / "family_shadow_nav.parquet",
                           filters=[("arm", "=", FactoryArm.C_ROLLING_REFIT_ROLLING_RECALIBRATION.value)])
    c_trades=pd.read_parquet(output_root / "family_shadow_trades.parquet",
                             filters=[("arm", "=", FactoryArm.C_ROLLING_REFIT_ROLLING_RECALIBRATION.value)])
    c_schedule=generation_schedule.loc[generation_schedule["arm"].eq(FactoryArm.C_ROLLING_REFIT_ROLLING_RECALIBRATION.value)]
    evidence=build_monthly_family_evidence(c_nav,c_schedule,matured_predictions=matured_predictions,trades=c_trades); evidence.to_parquet(output_root/"monthly_family_evidence.parquet",index=False)
    gate=run_gate1(evidence); gate.predictions.to_parquet(output_root/"gate1_predictions.parquet",index=False); gate.monthly_rank_metrics.to_csv(output_root/"gate1_monthly_rank_metrics.csv",index=False); gate.baseline_comparison.to_csv(output_root/"gate1_baseline_comparison.csv",index=False); _write_json(output_root/"gate1_block_bootstrap.json",gate.bootstrap)
    _write_json(output_root/"gate1_horizon_status.json",gate.horizon_status)
    gate1b=run_gate1b(evidence)
    downside_accepted={months:str(gate1b.horizon_status.get(months,"")).endswith("PASS") for months in (1,3)}
    gate2=run_gate2(evidence,include_downside=downside_accepted)
    gate3_panel=evidence
    if market_state is not None and not market_state.empty:
        market=market_state.copy(); market["assessment_date"]=pd.to_datetime(market["assessment_date"])
        gate3_panel=evidence.merge(market,on="assessment_date",how="left",validate="many_to_one")
    gate3=run_gate3(gate3_panel,include_downside=downside_accepted)
    active_generations=[]
    for (family_id,arm), group in generation_schedule.groupby(["family_id","arm"],sort=True):
        active=group.sort_values("activation_date").iloc[-1]
        active_generations.append({"family_id":family_id,"arm":arm,
                                   "generation_id":str(active.generation_id),
                                   "model_artifact_id":str(active.model_artifact_id),
                                   "activation_date":str(active.activation_date)})
    pre_holdout_state={"schema_version":"DYNAMIC_QBD_PRE_HOLDOUT_STATE_V1",
                       "as_of":str(max_nav_date),
                       "active_generations":active_generations,
                       "active_replay_states":replay_states,
                       "generation_schedule_hash":stable_hash(json.loads(generation_schedule.to_json(orient="records",date_format="iso"))),
                       "evidence_cursor":str(pd.to_datetime(evidence["assessment_date"]).max()) if not evidence.empty else None,
                       "gate_cursors":{"gate1":gate.status,"gate1b":gate1b.status,"gate2":gate2.status,"gate3":gate3.status}}
    pre_holdout_state["state_hash"]=stable_hash(pre_holdout_state)
    _write_json(output_root/"pre_holdout_state.json",pre_holdout_state)
    for name,result in (("gate1b",gate1b),("gate2",gate2),("gate3",gate3)):
        result.monthly_metrics.to_csv(output_root/f"{name}_monthly_metrics.csv",index=False)
        _write_json(output_root/f"{name}_block_bootstrap.json",result.bootstrap)
        _write_json(output_root/f"{name}_horizon_status.json",result.horizon_status)
    cost_columns=["assessment_date","target_months"]+[x for x in gate.baseline_comparison if x.startswith(("selector_excess_","incremental_vs_equal_")) and x.endswith("bps")]
    (gate.baseline_comparison[cost_columns] if set(cost_columns)<=set(gate.baseline_comparison) else pd.DataFrame(columns=cost_columns)).to_csv(output_root/"gate1_cost_stress.csv",index=False)
    scientific_support=bool(any(value.endswith("PASS") for value in gate.horizon_status.values()))
    summary={"status":"DYNAMIC_QBD_DEVELOPMENT_COMPLETE","abc":abc,"gate1_status":gate.status,
             "gate1_horizon_status":gate.horizon_status,"gate1b_status":gate1b.status,
             "gate2_status":gate2.status,"gate3_status":gate3.status,
             "shadow_selection_evidence_target_months":gate.bootstrap.get("selector_authority_target_months",[]),
             "selector_authority_target_months":[],
             "scientific_selection_evidence_supported":scientific_support,
             "router_mode":"SHADOW_RESEARCH_ONLY",
             "family_count":len(families),"holdout_contract":holdout_contract,
             "final_holdout_opened":False,"legacy_router_capital_authority":False,
             "capital_allocation_authority":False}
    summary["initial_pre_holdout_state_hash"]=pre_holdout_state["state_hash"]
    _write_json(output_root/"summary.json",summary)
    (output_root/"REPORT.md").write_text("# Dynamic QBD development\n\n- A/B/C complete: yes\n- Gate 1: `%s`\n- Router mode: `SHADOW_RESEARCH_ONLY`\n- Capital allocation authority: no\n- Final holdout opened: no\n"%gate.status,encoding="utf-8")
    structured = {
        "portfolio_paths": ("family_shadow_nav.parquet","family_shadow_trades.parquet","family_shadow_positions.parquet","family_relative_wealth.parquet"),
        "abc": ("abc_monthly_evidence.parquet","abc_arm_summary.csv","abc_decision.json"),
        "evidence": ("monthly_family_evidence.parquet","family_matured_outcomes.parquet","family_matured_predictions.parquet"),
        "gate1": ("gate1_predictions.parquet","gate1_monthly_rank_metrics.csv","gate1_baseline_comparison.csv","gate1_block_bootstrap.json","gate1_horizon_status.json"),
        "gate1b": ("gate1b_monthly_metrics.csv","gate1b_block_bootstrap.json","gate1b_horizon_status.json"),
        "gate2": ("gate2_monthly_metrics.csv","gate2_block_bootstrap.json","gate2_horizon_status.json"),
        "gate3": ("gate3_monthly_metrics.csv","gate3_block_bootstrap.json","gate3_horizon_status.json"),
    }
    for directory, names in structured.items():
        for name in names:
            source=output_root/name
            if source.is_file():
                _publish_structured_artifact(source,output_root/directory)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--family-registry",required=True); parser.add_argument("--predictions",required=True); parser.add_argument("--prices",required=True); parser.add_argument("--generation-schedule",required=True); parser.add_argument("--matured-predictions"); parser.add_argument("--output-root",required=True); parser.add_argument("--final-holdout-start",default="2026-07-25"); parser.add_argument("--holdout-contract",choices=("PRESERVE_HISTORICAL_LOCKBOX","PROSPECTIVE_FROM_2026_07_25"),required=True); args=parser.parse_args(argv)
    raw=json.loads(Path(args.family_registry).read_text(encoding="utf-8")); families=tuple(ExpertFamilySpec(**x) for x in raw["families"])
    run_development(families=families,predictions=pd.read_parquet(args.predictions),prices=pd.read_parquet(args.prices),generation_schedule=pd.read_parquet(args.generation_schedule),matured_predictions=pd.read_parquet(args.matured_predictions) if args.matured_predictions else None,output_root=args.output_root,final_holdout_start=date.fromisoformat(args.final_holdout_start),holdout_contract=args.holdout_contract)
    return 0


if __name__=="__main__": raise SystemExit(main())
