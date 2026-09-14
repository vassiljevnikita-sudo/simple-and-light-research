"""Fast contract regressions for the monthly recipe experiment."""
from __future__ import annotations

from datetime import date
import gzip
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from stock_predictor.data_processing.alpaca_direct_daily import materialize as materialize_direct_daily
from .dynamic_qbd_monthly_recipe_experiment import (
    MonthlyModelFactory, _attach_next_unseen_fold_evidence, _blend_frames, _hysteresis_accepts,
    _paired_bootstrap, _write_atomic_json, boundary_complete_segments,
)
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices


def _choice(candidate, family, values):
    folds=tuple(f"WF_{index:03d}_2020-01-01_2020-0{index+1}-28" for index in range(len(values)))
    return {"candidate_id":candidate,"family":family,"parameters":{},"fold_ids":folds,
            "fold_count":len(folds),"robust_score":float(np.mean(values))}


def main() -> int:
    incumbent=_choice("RIDGE","RIDGE",[.00,.01,.00,.01,.00])
    challenger=_choice("HGB","HIST_GRADIENT_BOOSTING",[.03,.04,.03,.04,.03])
    rows=[]
    for choice,values in ((incumbent,[.00,.01,.00,.01,.00]),(challenger,[.03,.04,.03,.04,.03])):
        for fold,value in zip(choice["fold_ids"],values):
            rows.append({"candidate_id":choice["candidate_id"],"family":choice["family"],"fold_id":fold,
                         "metrics":{"spearman":value}})
    bootstrap=_paired_bootstrap(incumbent=incumbent,challenger=challenger,rows=rows,cutoff=date(2020,6,30))
    assert bootstrap["probability_challenger_better"]>.99 and bootstrap["delta_ci_05"]>0
    assert not _hysteresis_accepts({"paired_fold_count":2,"probability_challenger_better":.999},.50), \
        "two folds incorrectly authorized a recipe switch"
    assert _hysteresis_accepts({"paired_fold_count":5,"probability_challenger_better":.95},.01)
    assert not _hysteresis_accepts({"paired_fold_count":5,"probability_challenger_better":.89},.01)
    assert not _hysteresis_accepts({"paired_fold_count":5,"probability_challenger_better":.95},.004)

    dates=pd.to_datetime(["2020-01-02"]*3+["2020-01-03"]*3)
    left=pd.DataFrame({"decision_date":dates,"ticker":["A","B","C"]*2,"score":[3,2,1,1,2,3],
                       "terminal_date":pd.to_datetime(["2020-01-06"]*3+["2020-01-07"]*3),
                       "observed_excess":[.1,.0,-.1,-.1,.0,.1]})
    right=left.copy(); right["score"]=[1,3,2,3,1,2]
    blend=_blend_frames(left,right,calibration=True)
    assert len(blend)==len(left),"50/50 blend became an intersection"
    assert np.allclose(blend["score"],.5*blend["ridge_percentile"]+.5*blend["hgb_percentile"])

    daily=pd.DataFrame({"arm":["R1"]*4,"recipe_candidate_id":["A","A","B","B"],
        "recipe_family":["RIDGE","RIDGE","HGB","HGB"],"date":pd.date_range("2020-01-01",periods=4),
        "recipe_boundary_day":[True,False,True,False],"daily_relative_return":[.01,.02,-.03,.04]})
    segments=boundary_complete_segments(daily)
    reconstructed=float(np.prod(1.0+segments["compounded_relative_return"].to_numpy(float))-1.0)
    expected=float(np.prod(1.0+daily["daily_relative_return"].to_numpy(float))-1.0)
    assert abs(reconstructed-expected)<1e-12 and int(segments["sessions"].sum())==len(daily)

    decisions=pd.DataFrame([{"winner_candidate_id":"HGB","winner_family":"HIST_GRADIENT_BOOSTING",
        "incumbent_candidate_id_before":"RIDGE","incumbent_family_before":"RIDGE",
        "available_fold_ids_json":'["WF_000"]'}])
    future_rows=[
        {"candidate_id":candidate,"family":family,"fold_id":fold,"metrics":{"spearman":value}}
        for candidate,family,fold,value in (
            ("HGB","HIST_GRADIENT_BOOSTING","INNER_FOR_WF_000",.99),
            ("RIDGE","RIDGE","INNER_FOR_WF_000",-.99),
            ("HGB","HIST_GRADIENT_BOOSTING","WF_001",.01),
            ("RIDGE","RIDGE","WF_001",.02))]
    unseen=_attach_next_unseen_fold_evidence(decisions,future_rows).iloc[0]
    assert unseen["next_unseen_fold_id"]=="WF_001"
    assert abs(unseen["next_unseen_challenger_minus_incumbent_spearman"]+.01)<1e-12

    # Cache reuse requires a complete generation manifest.  A single changed
    # artifact is a hard failure, not permission to reuse or silently retrain.
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); artifact=root/"fit"; artifact.mkdir()
        model=artifact/"model.joblib"; model.write_bytes(b"synthetic-model")
        calibration=artifact/"calibration-predictions.parquet"
        raw=artifact/"generation-predictions.parquet"; final=artifact/"model-predictions.parquet"
        frame=pd.DataFrame({"decision_date":[pd.Timestamp("2020-01-02")],"ticker":["A"],
                            "score":[.1],"model_artifact_id":["MODEL"]})
        frame.to_parquet(calibration,index=False); frame.drop(columns=["model_artifact_id"]).to_parquet(raw,index=False)
        frame.to_parquet(final,index=False)
        factory=object.__new__(MonthlyModelFactory)
        factory.git_sha="a"*40; factory.training_source_sha256="b"*64; factory.candidate_metrics_sha256="c"*64
        family=type("Family",(),{"family_hash":"FAMILY_HASH"})()
        choice={"candidate_id":"RIDGE","family":"RIDGE","parameters":{"alpha":1.0}}
        payload=factory._generation_manifest_payload(family=family,choice=choice,cutoff=date(2020,1,31),
            build={"dataset_fingerprint":"DATASET_HASH","model_artifact_id":"MODEL"},calibration_path=calibration,
            raw_prediction_path=raw,prediction_path=final,model_path=model)
        _write_atomic_json(artifact/"generation-manifest.json",payload)
        verified=factory._verify_generation_manifest(artifact_root=artifact,family=family,choice=choice,
            cutoff=date(2020,1,31),calibration_path=calibration,raw_prediction_path=raw,
            prediction_path=final,model_path=model)
        assert verified["model_artifact_id"]=="MODEL"
        raw.write_bytes(b"mutated")
        try:
            factory._verify_generation_manifest(artifact_root=artifact,family=family,choice=choice,
                cutoff=date(2020,1,31),calibration_path=calibration,raw_prediction_path=raw,
                prediction_path=final,model_path=model)
            raise AssertionError("mutated monthly artifact was reused")
        except RuntimeError as exc:
            assert str(exc).startswith("MONTHLY_MODEL_CACHE_ARTIFACT_HASH_MISMATCH")

    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); raw=root/"URTH.jsonl.gz"; done=root/"URTH.done.json"; output=root/"URTH.parquet"
        row={"ticker":"URTH","timestamp_ms":1451865600000,"open":68.0,"high":69.0,"low":67.0,
             "close":68.5,"volume":1000,"vwap":68.3,"transactions":10}
        with gzip.open(raw,"wt",encoding="utf-8") as handle:
            handle.write(json.dumps(row)+"\n")
        done.write_text(json.dumps({"timeframe":"1Day","feed":"sip","adjustment":"split",
                                    "ticker":"URTH","start":"2016-01-04","end":"2016-01-04"}),encoding="utf-8")
        materialize_direct_daily(raw_path=raw,output_path=output)
        direct=pd.read_parquet(output)
        manifest=json.loads(output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8"))
        assert len(direct)==1 and direct.loc[0,"price_boundary_contract"]=="ALPACA_SIP_1DAY_DIRECT"
        assert manifest["timeframe"]=="1Day" and manifest["feed"]=="sip" and manifest["rows"]==1

    # A direct daily stock fallback may repair only a boundary that has failed
    # the minute-derived timestamp contract. A valid primary boundary must not
    # be silently replaced by a different provider value.
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); store=root/"daily"; fallback_root=root/"fallback"
        session=pd.Timestamp("2020-01-02")
        panel=root/"panel.parquet"
        pd.DataFrame({"decision_date":[session],"ticker":["AAA"]}).to_parquet(panel,index=False)
        for ticker,frame in {
            "AAA":pd.DataFrame({"session_date":[session],"ticker":["AAA"],"open":[1.0],"close":[2.0],
                "volume":[100.0],"minute_count":[380],
                "first_timestamp_ms":[int(pd.Timestamp("2020-01-02 14:40:00Z").timestamp()*1000)],
                "last_timestamp_ms":[int(pd.Timestamp("2020-01-02 20:59:00Z").timestamp()*1000)]}),
            "URTH":pd.DataFrame({"session_date":[session],"ticker":["URTH"],"open":[50.0],"close":[51.0]})
        }.items():
            target=store/"schema=v1"/"source=alpaca"/f"ticker={ticker}"/"variant=sip_split"/"bars.parquet"
            target.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(target,index=False)
        def write_direct(ticker: str, open_value: float, close_value: float, output_path: Path) -> Path:
            raw=root/f"{ticker}.jsonl.gz"; done=root/f"{ticker}.done.json"
            row={"ticker":ticker,"timestamp_ms":int(pd.Timestamp("2020-01-02 05:00:00Z").timestamp()*1000),
                 "open":open_value,"high":max(open_value,close_value),"low":min(open_value,close_value),
                 "close":close_value,"volume":1000,"vwap":close_value,"transactions":10}
            with gzip.open(raw,"wt",encoding="utf-8") as handle: handle.write(json.dumps(row)+"\n")
            done.write_text(json.dumps({"timeframe":"1Day","feed":"sip","adjustment":"split",
                "ticker":ticker,"start":"2020-01-02","end":"2020-01-02"}),encoding="utf-8")
            return materialize_direct_daily(raw_path=raw,output_path=output_path)
        benchmark=write_direct("URTH",50.0,51.0,root/"URTH.parquet")
        stock=write_direct("AAA",10.0,20.0,fallback_root/"ticker=AAA"/"bars.parquet")
        stock_manifest=stock.with_suffix(stock.suffix+".manifest.json")
        payload=json.loads(stock_manifest.read_text(encoding="utf-8"))
        payload["usage"]="ROW_LEVEL_FALLBACK_FOR_INVALID_MINUTE_EXECUTION_BOUNDARIES"
        stock_manifest.write_text(json.dumps(payload),encoding="utf-8")
        (fallback_root/"manifest.json").write_text(json.dumps({"schema_version":"ALPACA_DIRECT_DAILY_TREE_V1",
            "ticker_count":1,"tickers":["AAA"],
            "usage":"ROW_LEVEL_FALLBACK_FOR_INVALID_MINUTE_EXECUTION_BOUNDARIES"}),encoding="utf-8")
        projected=materialize_daily_store_prices(daily_store_root=store,signal_panel=panel,
            start=date(2020,1,2),end=date(2020,1,2),output_path=root/"prices.parquet",
            benchmark_daily_path=benchmark,direct_daily_stock_root=fallback_root)
        aaa=pd.read_parquet(projected).loc[lambda x:x["ticker"].eq("AAA")].iloc[0]
        assert aaa["open"]==10.0 and aaa["close"]==2.0
        assert aaa["open_provider_contract"]=="ALPACA_SIP_1DAY_INVALID_MINUTE_FALLBACK"
        assert aaa["close_provider_contract"]=="MINUTE_DERIVED_DAILY"
    print("DYNAMIC_QBD_MONTHLY_RECIPE_SELF_TEST_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
