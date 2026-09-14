from __future__ import annotations
import json
import re
from pathlib import Path
from .top10_qbd_router_contracts import canonical_json, sha256_fingerprint
from .top10_qbd_expert_registry import registry_hash

MANIFEST_NAME='QBD_ROUTER_V1_FROZEN_MANIFEST.json'
def build_manifest(*, policy, registry, code_commit, data_fingerprint, evaluation_contract_hash, holdout_start=None, holdout_end=None):
    return {'contract_id':policy.contract_id,'code_commit':code_commit,'policy_hash':policy.policy_hash,'registry_hash':registry_hash(registry),'expert_artifact_hashes':{x.expert_id:x.model_artifact_hash for x in registry if x.model_artifact_hash},'feature_schema_hashes':{x.expert_id:x.feature_schema_hash for x in registry if x.feature_schema_hash},'adaptation_controller_hashes':{x.expert_id:x.adaptation_controller_hash for x in registry if x.adaptation_controller_hash},'cost_policy':{'stock_roundtrip_bps':policy.stock_roundtrip_bps,'benchmark_roundtrip_bps':policy.benchmark_roundtrip_bps,'switch_cost_bps':policy.switch_cost_bps},'data_fingerprint':data_fingerprint,'evaluation_contract_hash':evaluation_contract_hash,'final_holdout':{'start':(holdout_start or policy.final_holdout_start).isoformat(),'end':(holdout_end or policy.final_holdout_end).isoformat()}}
def write_manifest(path, manifest):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(canonical_json(manifest)+'\n',encoding='utf-8'); return sha256_fingerprint(manifest)
def validate_manifest(path, *, policy, registry, code_commit, evaluation_contract_hash):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    checks={'contract_id':raw.get('contract_id')==policy.contract_id,'code_commit':raw.get('code_commit')==code_commit,'policy_hash':raw.get('policy_hash')==policy.policy_hash,'registry_hash':raw.get('registry_hash')==registry_hash(registry),'evaluation_contract_hash':raw.get('evaluation_contract_hash')==evaluation_contract_hash}
    required_hashes = list(raw.get('expert_artifact_hashes', {}).values()) + list(raw.get('feature_schema_hashes', {}).values()) + list(raw.get('adaptation_controller_hashes', {}).values())
    if any((not isinstance(x, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', x) or x.upper().startswith('UNRESOLVED')) for x in required_hashes):
        raise ValueError('frozen manifest contains unresolved artifact identity')
    if not raw.get('data_fingerprint') or not raw.get('final_holdout', {}).get('start') or not raw.get('final_holdout', {}).get('end'):
        raise ValueError('frozen manifest is missing data fingerprint or holdout boundaries')
    if not all(checks.values()): raise ValueError('frozen manifest mismatch: '+','.join(k for k,v in checks.items() if not v))
    return raw
