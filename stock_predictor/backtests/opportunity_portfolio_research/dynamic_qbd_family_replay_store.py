"""Deterministic replay checkpoint with prefix-integrity verification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .contract_fingerprints import stable_hash
from .dynamic_qbd_portfolio_replay import replay_family


def _records(frame):
    if frame is None or frame.empty: return []
    return json.loads(frame.to_json(orient="records",date_format="iso"))


class FamilyReplayCheckpointStore:
    def __init__(self,path): self.path=Path(path)
    def load(self):
        if not self.path.exists(): return None
        payload=json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("replay_state_hash") != stable_hash(payload.get("replay_state")):
            raise ValueError("REPLAY_CHECKPOINT_STATE_HASH_MISMATCH")
        return payload
    def save(self,payload):
        self.path.parent.mkdir(parents=True,exist_ok=True); temp=self.path.with_name(self.path.name+f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload,sort_keys=True,indent=2,default=str)+"\n",encoding="utf-8"); os.replace(temp,self.path)


def run_checkpointed_family_replay(*, checkpoint_store, end, **kwargs):
    prior=checkpoint_store.load()
    resume_state=prior.get("replay_state") if prior else None
    result=replay_family(end=end,resume_state=resume_state,**kwargs)
    curve_records=_records(result.get("curve")); trades=result.get("trades",[]); open_positions=result.get("open_positions",[])
    if prior:
        prior_end=pd.Timestamp(prior["as_of"])
        prefix=[x for x in curve_records if pd.Timestamp(x["date"])<=prior_end]
        if stable_hash(prefix)!=prior["curve_prefix_hash"] or stable_hash([x for x in trades if pd.Timestamp(x["exit_date"])<=prior_end])!=prior["trade_prefix_hash"]:
            raise AssertionError("CONTINUOUS_RESTART_PREFIX_MISMATCH")
    replay_state=result.get("replay_state")
    if not replay_state:
        raise AssertionError("REPLAY_DID_NOT_RETURN_RESUMABLE_STATE")
    payload={"schema_version":"DYNAMIC_QBD_REPLAY_CHECKPOINT_V2","as_of":str(pd.Timestamp(replay_state["as_of"])),
             "curve_prefix_hash":stable_hash(curve_records),"trade_prefix_hash":stable_hash(trades),
             "open_positions":open_positions,"open_position_lineage":{x["ticker"]:{k:x.get(k) for k in ("family_id","generation_id","entry_policy_id","exit_policy_id")} for x in open_positions},
             "generation_schedule_hash":stable_hash(_records(kwargs["generation_schedule"])),
             "replay_state":replay_state,"replay_state_hash":stable_hash(replay_state)}
    checkpoint_store.save(payload); return result
