from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict
import math
from threading import Lock
import weakref
import pandas as pd
import numpy as np

from .portfolio_allocation_weights import requested_notionals
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .german_retail_tax_engine import TaxLedger, benchmark_tax_approximate

_PRICE_CACHE: dict[int, tuple[dict, dict, tuple[pd.Timestamp, ...]]] = {}
_PRICE_CACHE_LOCK = Lock()
_SIGNAL_CACHE: dict[int, tuple[weakref.ReferenceType, dict]] = {}
_SIGNAL_CACHE_LOCK = Lock()
_ACCOUNTING_TOL = 1e-7


def prepare_prices(prices: pd.DataFrame) -> tuple[dict, dict, tuple[pd.Timestamp, ...]]:
    """Build the immutable price lookup once per price panel."""
    cache_key = id(prices)
    cached = _PRICE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _PRICE_CACHE_LOCK:
        cached = _PRICE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        urth0 = prices.loc[prices["ticker"].eq("URTH"), ["date", "open", "close"]].sort_values("date")
        umap0 = {pd.Timestamp(r.date): (float(r.open), float(r.close)) for r in urth0.itertuples(index=False)}
        pmap0 = {
            (pd.Timestamp(r.date), str(r.ticker)): (float(r.open), float(r.close))
            for r in prices[["date", "ticker", "open", "close"]].itertuples(index=False)
        }
        all_dates = tuple(sorted(umap0))
        cached = (pmap0, umap0, all_dates)
        _PRICE_CACHE[cache_key] = cached
        return cached


def _build_prepared_signals(signals: pd.DataFrame) -> dict:
    if signals.empty:
        return {"by_date": {}, "top_score_by_date": {}, "row_count": 0}

    s = signals.loc[:, ["decision_date", "ticker", "score"]].copy()
    s["decision_date"] = pd.to_datetime(s["decision_date"])
    s = s.loc[s["decision_date"].notna()].sort_values(
        ["decision_date", "score"], ascending=[True, False], na_position="last"
    )
    by_date: dict[pd.Timestamp, tuple[tuple[str, ...], np.ndarray, np.ndarray, int]] = {}
    top_score_by_date: dict[pd.Timestamp, float] = {}
    for d, g in s.groupby("decision_date", sort=False):
        scores_all = g["score"].to_numpy(dtype=float, copy=True)
        finite_count = int(np.isfinite(scores_all).sum())
        if finite_count:
            scores = scores_all[:finite_count]
            tickers = tuple(g["ticker"].astype(str).iloc[:finite_count].tolist())
            neg_scores = -scores
            top_score_by_date[pd.Timestamp(d)] = float(scores[0])
        else:
            scores = np.empty(0, dtype=float)
            neg_scores = np.empty(0, dtype=float)
            tickers = tuple()
        by_date[pd.Timestamp(d)] = (tickers, scores, neg_scores, int(len(g)))
    return {
        "by_date": by_date,
        "top_score_by_date": top_score_by_date,
        "row_count": int(len(s)),
    }


def prepare_signals(signals: pd.DataFrame) -> dict:
    """Prepare one signal frame once and reuse it while that DataFrame is alive."""
    key = id(signals)
    with _SIGNAL_CACHE_LOCK:
        cached = _SIGNAL_CACHE.get(key)
        if cached is not None and cached[0]() is signals:
            return cached[1]

    prepared = _build_prepared_signals(signals)

    def _drop(ref, cache_key=key):
        with _SIGNAL_CACHE_LOCK:
            current = _SIGNAL_CACHE.get(cache_key)
            if current is not None and current[0] is ref:
                _SIGNAL_CACHE.pop(cache_key, None)

    ref = weakref.ref(signals, _drop)
    with _SIGNAL_CACHE_LOCK:
        current = _SIGNAL_CACHE.get(key)
        if current is not None and current[0]() is signals:
            return current[1]
        _SIGNAL_CACHE[key] = (ref, prepared)
    return prepared


def _annualized_return(initial: float, terminal: float, calendar_days: int) -> float:
    years = max(calendar_days / 365.25, 1 / 365.25)
    return (terminal / initial) ** (1 / years) - 1 if initial > 0 and terminal > 0 else -1.0


def _distribution_contract(distributions: pd.DataFrame | None, all_dates: tuple[pd.Timestamp,...]) -> tuple[dict,dict]:
    if distributions is None or distributions.empty:
        return {},{}
    required={"ticker","ex_date","payable_date","cash_amount"}
    missing=required-set(distributions)
    if missing: raise ValueError(f"DISTRIBUTION_COLUMNS_MISSING:{sorted(missing)}")
    frame=distributions[list(required)].copy()
    frame["ticker"]=frame["ticker"].astype(str)
    frame["ex_date"]=pd.to_datetime(frame["ex_date"]).dt.normalize()
    frame["payable_date"]=pd.to_datetime(frame["payable_date"]).dt.normalize()
    frame["cash_amount"]=pd.to_numeric(frame["cash_amount"],errors="raise").astype(float)
    if frame.duplicated(["ticker","ex_date","payable_date"]).any(): raise ValueError("DUPLICATE_DISTRIBUTION_EVENT")
    if (frame["cash_amount"]<=0).any() or (frame["payable_date"]<frame["ex_date"]).any():
        raise ValueError("INVALID_DISTRIBUTION_EVENT")
    sessions=pd.DatetimeIndex(all_dates)
    ex_map={}; payable_map={}
    for row in frame.itertuples(index=False):
        ex=pd.Timestamp(row.ex_date); payable=pd.Timestamp(row.payable_date)
        if ex<sessions[0] or ex>sessions[-1]: continue
        if ex not in sessions: raise ValueError(f"DISTRIBUTION_EX_DATE_NOT_SESSION:{row.ticker}:{ex.date()}")
        pay_index=int(sessions.searchsorted(payable,side="left"))
        if pay_index>=len(sessions): continue
        payment_session=pd.Timestamp(sessions[pay_index])
        event={"ticker":str(row.ticker),"ex_date":ex,"payable_date":payable,
               "payment_session":payment_session,"cash_amount":float(row.cash_amount)}
        ex_map.setdefault(ex,[]).append(event); payable_map.setdefault(payment_session,[]).append(event)
    return ex_map,payable_map


def _benchmark_total_return_maps(*, umap: dict, all_dates: tuple[pd.Timestamp,...], start_date: pd.Timestamp,
                                 distribution_ex_map: dict) -> tuple[dict,dict,dict]:
    # A distribution becomes an asset on its ex-date, not only when cash is
    # paid.  Keeping the receivable in the index avoids a false NAV trough at
    # ex-date and a matching artificial jump on the payable session.
    units=1.0; receivables=defaultdict(float); receivable_balance=0.0; opens={}; closes={}; paid={}
    start_index=bisect_left(all_dates,pd.Timestamp(start_date))
    for d in all_dates[start_index:]:
        if d != start_date:
            for event in distribution_ex_map.get(d,[]):
                if event["ticker"]=="URTH":
                    entitlement=units*event["cash_amount"]
                    receivables[event["payment_session"]] += entitlement
                    receivable_balance += entitlement
        payout=float(receivables.pop(d,0.0)); paid[d]=payout
        if payout:
            receivable_balance -= payout
            units += payout/umap[d][0]
        opens[d]=units*umap[d][0]+receivable_balance
        closes[d]=units*umap[d][1]+receivable_balance
    return opens,closes,paid


def _metrics(
    curve: pd.DataFrame,
    initial: float,
    trades: list[dict],
    costs: dict,
    tax: TaxLedger,
    benchmark_initial: float,
    benchmark_terminal_after_tax: float | None = None,
    benchmark_tax: float = 0.0,
    strategy_terminal_after_tax: float | None = None,
) -> dict:
    strategy = curve["strategy_value"].astype(float)
    bench_curve = curve["urth_value"].astype(float)
    peak = strategy.cummax()
    dd = strategy / peak - 1.0
    bpeak = bench_curve.cummax()
    bdd = bench_curve / bpeak - 1.0
    terminal_pre_tax = float(strategy.iloc[-1])
    terminal = float(strategy_terminal_after_tax if strategy_terminal_after_tax is not None else terminal_pre_tax)
    bench_pre_tax_terminal = float(bench_curve.iloc[-1])
    bench_terminal = float(benchmark_terminal_after_tax if benchmark_terminal_after_tax is not None else bench_pre_tax_terminal)
    calendar_days = int((curve["date"].iloc[-1] - curve["date"].iloc[0]).days)
    daily = strategy.pct_change().dropna()
    b_daily = bench_curve.pct_change().dropna()
    trade_excess = [float(x.get("excess_return", 0.0)) for x in trades]
    wins = [x for x in trade_excess if x > 0]
    cagr = _annualized_return(initial, terminal, calendar_days)
    urth_cagr = _annualized_return(benchmark_initial, bench_terminal, calendar_days)
    return {
        "initial_value": initial,
        "terminal_value": terminal,
        "terminal_value_pre_tax": terminal_pre_tax,
        "urth_terminal_value": bench_terminal,
        "urth_terminal_value_pre_tax": bench_pre_tax_terminal,
        "total_return": terminal / initial - 1,
        "urth_total_return": bench_terminal / benchmark_initial - 1,
        "cagr": cagr,
        "urth_cagr": urth_cagr,
        "cagr_excess": cagr - urth_cagr,
        "terminal_wealth_excess_eur": terminal - bench_terminal,
        "total_return_excess": terminal / initial - bench_terminal / benchmark_initial,
        "max_drawdown": float(dd.min()),
        "urth_max_drawdown": float(bdd.min()),
        "worst_relative_drawdown": float((strategy / bench_curve / (strategy / bench_curve).cummax() - 1).min()),
        "annualized_volatility": float(daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 else 0.0,
        "urth_annualized_volatility": float(b_daily.std(ddof=1) * math.sqrt(252)) if len(b_daily) > 1 else 0.0,
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trade_excess) if trade_excess else 0.0,
        "excess_hit_rate": len(wins) / len(trade_excess) if trade_excess else 0.0,
        "mean_trade_return": float(np.mean([x.get("stock_return", 0.0) for x in trades])) if trades else 0.0,
        "median_trade_return": float(np.median([x.get("stock_return", 0.0) for x in trades])) if trades else 0.0,
        "mean_trade_excess": float(np.mean(trade_excess)) if trades else 0.0,
        "median_trade_excess": float(np.median(trade_excess)) if trades else 0.0,
        "mean_holding_days": float(np.mean([x.get("holding_days", 0) for x in trades])) if trades else 0.0,
        "median_holding_days": float(np.median([x.get("holding_days", 0) for x in trades])) if trades else 0.0,
        "turnover": float(sum(x.get("buy_notional", 0.0) + x.get("sell_notional", 0.0) for x in trades) / max(initial, 1.0)),
        "stock_exposure": float(curve["stock_exposure"].mean()),
        "urth_exposure": float(curve["urth_exposure"].mean()),
        "cash_exposure": float(curve["cash_exposure"].mean()),
        "average_positions": float(curve["positions"].mean()),
        "max_positions": int(curve["positions"].max()),
        "max_abs_accounting_error_eur": float(curve["accounting_error_eur"].abs().max()),
        "max_total_exposure": float((curve["stock_exposure"] + curve["urth_exposure"] + curve["cash_exposure"]
                                      + curve.get("distribution_receivable_exposure",0.0)).max()),
        "transaction_cost_eur": float(costs["transaction_cost_eur"]),
        "fx_cost_eur": 0.0,
        "total_cost_eur": float(costs["transaction_cost_eur"]),
        "cost_drag_pct": float(costs["transaction_cost_eur"] / max(initial, 1.0)),
        "tax_paid": float(tax.tax_paid),
        "tax_drag": float(tax.tax_paid / max(initial, 1.0)),
        "realized_gain": tax.realized_gain,
        "realized_loss": tax.realized_loss,
        "used_sparer_pauschbetrag": tax.used_allowance,
        "remaining_loss_carryforward": tax.loss_carryforward,
        "benchmark_tax_approximate": bool(benchmark_tax),
        "benchmark_tax_paid": float(benchmark_tax),
        "period_days": calendar_days,
    }


def replay(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    policy: Policy,
    cost: CostModel,
    tax_config: TaxConfig,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    initial: float = 10000.0,
    regime: pd.DataFrame | None = None,
    resolved_threshold: float | None = None,
    prepared_signals: dict | None = None,
    generation_schedule: pd.DataFrame | None = None,
    resume_state: dict | None = None,
    distributions: pd.DataFrame | None = None,
) -> dict:
    pmap, umap, all_dates = prepare_prices(prices)
    original_lo = 0 if start is None else bisect_left(all_dates, pd.Timestamp(start))
    lo = original_lo
    if resume_state is not None:
        lo = bisect_right(all_dates, pd.Timestamp(resume_state["as_of"]))
    hi = len(all_dates) if end is None else bisect_right(all_dates, pd.Timestamp(end))
    dates_tuple = all_dates[lo:hi]
    if not dates_tuple and resume_state is None:
        return {
            "metrics": {"trade_count": 0, "initial_value": initial, "terminal_value": initial},
            "curve": pd.DataFrame(),
            "trades": [],
            "cost_audit": {},
            "tax_audit": {},
        }

    dates = pd.DatetimeIndex(dates_tuple)
    quality_enforced={"open_quality_ok","close_quality_ok"} <= set(prices)
    quality={}
    if quality_enforced:
        quality={(pd.Timestamp(row.date),str(row.ticker)):(bool(row.open_quality_ok),bool(row.close_quality_ok))
                 for row in prices[["date","ticker","open_quality_ok","close_quality_ok"]].itertuples(index=False)}

    def require_price_quality(d, ticker, field):
        if not quality_enforced: return
        flags=quality.get((pd.Timestamp(d),str(ticker)))
        index=0 if field=="open" else 1
        if flags is None or not flags[index]:
            raise ValueError(f"REPLAY_{field.upper()}_PRICE_BOUNDARY_INVALID:{ticker}:{pd.Timestamp(d).date()}")

    for benchmark_date in dates:
        require_price_quality(benchmark_date,"URTH","close")
    idx = {d: i for i, d in enumerate(all_dates)}
    distribution_ex_map,_distribution_payable_map=_distribution_contract(distributions,all_dates)
    benchmark_start_date=(pd.Timestamp(resume_state["benchmark_start_date"]) if resume_state is not None
                          else all_dates[original_lo])
    benchmark_open_index,benchmark_close_index,benchmark_distribution_paid=_benchmark_total_return_maps(
        umap=umap,all_dates=all_dates,start_date=benchmark_start_date,distribution_ex_map=distribution_ex_map)
    benchmark_initial_index_close=float(benchmark_close_index[benchmark_start_date])
    prepared = prepared_signals if prepared_signals is not None else prepare_signals(signals)
    by_date = prepared["by_date"]
    top_score_by_date = prepared["top_score_by_date"]
    if resume_state is not None and "threshold" in resume_state:
        threshold = float(resume_state["threshold"])
    elif resolved_threshold is not None:
        threshold = float(resolved_threshold)
    else:
        top_scores = [top_score_by_date[d] for d in dates if d in top_score_by_date]
        threshold = float(np.quantile(top_scores, policy.score_quantile)) if top_scores else float("inf")

    lineage_schedule = []
    if generation_schedule is not None and not generation_schedule.empty:
        required = {"activation_date", "family_id", "generation_id", "resolved_threshold", "entry_policy_id", "exit_policy_id"}
        missing = required - set(generation_schedule)
        if missing:
            raise ValueError(f"GENERATION_SCHEDULE_COLUMNS_MISSING:{sorted(missing)}")
        for row in generation_schedule.sort_values("activation_date").itertuples(index=False):
            lineage_schedule.append((pd.Timestamp(row.activation_date), {
                "family_id": str(row.family_id), "generation_id": str(row.generation_id),
                "model_artifact_id": str(getattr(row, "model_artifact_id", "")),
                "resolved_threshold": float(row.resolved_threshold), "entry_policy_id": str(row.entry_policy_id),
                "exit_policy_id": str(row.exit_policy_id),
                "exit_generation_id": str(getattr(row, "exit_generation_id", row.exit_policy_id)),
                "resolved_top_fraction": float(getattr(row, "resolved_top_fraction", policy.top_fraction)),
            }))

    def generation_as_of(d):
        active = None
        for activation, record in lineage_schedule:
            if activation > d:
                break
            active = record
        return active

    if resume_state is None:
        benchmark_initial_close=umap[benchmark_start_date][1]
        pending=defaultdict(list); positions={}; cash=0.0
        urth_units=initial/benchmark_initial_close; tax=TaxLedger(tax_config)
        trades=[]; costs={"transaction_cost_eur":0.0}; curve=[]
        sleeve_breach_count=0; entry_scores={}; pending_distributions=defaultdict(list)
        distribution_audit={"gross_distribution_eur":0.0,"distribution_tax_eur":0.0,
                            "distribution_event_count":0,"benchmark_gross_distribution_eur":0.0}
    else:
        if float(resume_state["initial_value"]) != float(initial):
            raise ValueError("REPLAY_RESUME_INITIAL_VALUE_MISMATCH")
        benchmark_initial_close=float(resume_state["benchmark_initial_close"])
        pending=defaultdict(list,{pd.Timestamp(key):list(value) for key,value in resume_state.get("pending_orders",{}).items()})
        positions={}
        for ticker,raw in resume_state.get("positions",{}).items():
            position=dict(raw); position["entry_date"]=pd.Timestamp(position["entry_date"])
            positions[str(ticker)]=position
        cash=float(resume_state["cash"]); urth_units=float(resume_state["urth_units"])
        tax=TaxLedger.from_snapshot(tax_config,resume_state.get("tax_ledger"))
        trades=list(resume_state.get("trades",[]))
        for trade in trades:
            trade["entry_date"]=pd.Timestamp(trade["entry_date"]); trade["exit_date"]=pd.Timestamp(trade["exit_date"])
        costs={"transaction_cost_eur":float(resume_state.get("transaction_cost_eur",0.0))}
        curve=list(resume_state.get("curve",[]))
        for point in curve: point["date"]=pd.Timestamp(point["date"])
        sleeve_breach_count=int(resume_state.get("sleeve_breach_count",0))
        entry_scores={str(key):float(value) for key,value in resume_state.get("entry_scores",{}).items()}
        pending_distributions=defaultdict(list,{pd.Timestamp(key):list(value)
                                                for key,value in resume_state.get("pending_distributions",{}).items()})
        distribution_audit={key:float(value) for key,value in resume_state.get("distribution_audit",{}).items()}
        for key in ("gross_distribution_eur","distribution_tax_eur","distribution_event_count",
                    "benchmark_gross_distribution_eur"):
            distribution_audit.setdefault(key,0.0)

    def distribution_receivable_value() -> float:
        return float(sum(float(payment["gross_cash"]) for payments in pending_distributions.values()
                         for payment in payments))

    def components(d, px="close") -> tuple[float, float, float, float, float]:
        require_price_quality(d,"URTH",px)
        u = umap[d][1 if px == "close" else 0]
        urth_value = urth_units * u
        stock_value = 0.0
        for t, pos in positions.items():
            q = pmap.get((d, t))
            if q:
                require_price_quality(d,t,px)
                stock_value += pos["qty"] * q[1 if px == "close" else 0]
        receivable_value=distribution_receivable_value()
        total = cash + urth_value + stock_value + receivable_value
        return total, cash, urth_value, stock_value, receivable_value

    def buy_urth(notional, d):
        nonlocal urth_units, cash
        require_price_quality(d,"URTH","open")
        u = umap[d][0]
        take = min(max(0.0, notional), max(0.0, cash))
        urth_units += take / u
        cash -= take

    def sell_urth(notional, d):
        nonlocal urth_units, cash
        require_price_quality(d,"URTH","open")
        u = umap[d][0]
        take = min(max(0.0, notional), max(0.0, urth_units * u))
        urth_units -= take / u
        cash += take

    def sell_stock(t, d):
        nonlocal cash
        pos = positions.pop(t, None)
        if not pos or (d, t) not in pmap:
            return
        require_price_quality(d,t,"open"); require_price_quality(d,"URTH","open")
        px = pmap[(d, t)][0]
        gross = pos["qty"] * px
        sell_fee = gross * cost.per_side_bps / 10000
        net = gross - sell_fee
        cash += net
        costs["transaction_cost_eur"] += sell_fee
        tax_paid = tax.realize_stock_trade(d.date(), gross, pos["tax_basis"], sell_fee)
        cash -= tax_paid
        entry_benchmark=float(pos.get("entry_benchmark_index",pos["entry_urth"]))
        excess = px / pos["entry_price"] / (benchmark_open_index[d] / entry_benchmark) - 1
        trades.append(
            {
                "ticker": t,
                "entry_date": pos["entry_date"],
                "exit_date": d,
                "holding_days": idx[d] - pos["entry_index"],
                "stock_return": px / pos["entry_price"] - 1,
                "excess_return": excess,
                "buy_notional": pos["buy_notional"],
                "sell_notional": gross,
                "cost_eur": pos["buy_fee"] + sell_fee,
                "tax_eur": tax_paid,
                "family_id": pos.get("family_id"),
                "generation_id": pos.get("generation_id"),
                "entry_family_id": pos.get("family_id"),
                "entry_generation_id": pos.get("generation_id"),
                "entry_model_artifact_id": pos.get("model_artifact_id"),
                "entry_policy_id": pos.get("entry_policy_id"),
                "entry_score": pos.get("entry_score"),
                "entry_score_rank": pos.get("entry_score_rank"),
                "entry_threshold": pos.get("entry_threshold"),
                "entry_threshold_distance": pos.get("entry_threshold_distance"),
                "quantity": pos.get("qty"),
                "entry_price": pos.get("entry_price"),
                "entry_urth": pos.get("entry_urth"),
                "exit_policy_id": pos.get("exit_policy_id"),
                "exit_generation_id": pos.get("exit_generation_id"),
            }
        )
        buy_urth(max(0.0, cash), d)

    for d in dates:
        next_d=all_dates[idx[d]+1] if idx[d]+1<len(all_dates) else None
        if resume_state is not None or d != benchmark_start_date:
            for event in distribution_ex_map.get(d,[]):
                units=(urth_units if event["ticker"]=="URTH" else
                       float(positions.get(event["ticker"],{}).get("qty",0.0)))
                if units>0:
                    pending_distributions[event["payment_session"]].append({
                        **event,"entitled_units":units,"gross_cash":units*event["cash_amount"]})
        strategy_distribution_cashflow=0.0
        for payment in pending_distributions.pop(d,[]):
            gross=float(payment["gross_cash"])
            paid_tax=tax.realize_cash_distribution(d.date(),gross)
            net=gross-paid_tax; cash += net; strategy_distribution_cashflow += net
            distribution_audit["gross_distribution_eur"] += gross
            distribution_audit["distribution_tax_eur"] += paid_tax
            distribution_audit["distribution_event_count"] += 1
            buy_urth(net,d)
        benchmark_cashflow=float(benchmark_distribution_paid.get(d,0.0))*initial/benchmark_initial_index_close
        distribution_audit["benchmark_gross_distribution_eur"] += benchmark_cashflow
        active_generation = generation_as_of(d)
        current_threshold = float(active_generation["resolved_threshold"]) if active_generation else threshold
        # Persist the threshold actually authoritative for the latest
        # processed session.  Without this, a resumed segment retained the
        # first segment's bootstrap threshold while a one-shot replay ended
        # with a different schedule-derived value.
        threshold = current_threshold
        current_top_fraction = float(active_generation["resolved_top_fraction"]) if active_generation else policy.top_fraction
        actions = pending.pop(d, [])
        for action in actions:
            if action["kind"] == "sell":
                sell_stock(action["ticker"], d)

        buy_actions = [a for a in actions if a["kind"] == "buy" and a["ticker"] not in positions and (d, a["ticker"]) in pmap]
        if buy_actions:
            equity_open, _, _, stock_open, _ = components(d, "open")
            sleeve_capacity = max(0.0, equity_open * policy.sleeve - stock_open)
            if policy.allocation == "EQUAL_ACTIVE":
                requested = [sleeve_capacity / len(buy_actions)] * len(buy_actions) if buy_actions else []
            elif policy.allocation.startswith(("RANK_POWER:", "SCORE_EXCESS_POWER:", "SCORE_SOFTMAX:")):
                requested = requested_notionals(
                    policy.allocation,
                    [float(a.get("score", 0.0)) for a in buy_actions],
                    sleeve_capacity,
                    threshold=current_threshold,
                )
            else:
                slot = equity_open * policy.sleeve / max(policy.max_names, 1)
                requested = [slot] * len(buy_actions)
            for action, requested_notional in zip(buy_actions, requested):
                equity_open, _, _, stock_open, _ = components(d, "open")
                remaining_capacity = max(0.0, equity_open * policy.sleeve - stock_open)
                target = min(max(0.0, requested_notional), remaining_capacity)
                if target <= 0.0:
                    continue
                sell_urth(target, d)
                require_price_quality(d,action["ticker"],"open")
                px = pmap[(d, action["ticker"])][0]
                buy_fee = target * cost.per_side_bps / 10000
                buy_notional = max(0.0, target - buy_fee)
                if buy_notional <= 0.0:
                    buy_urth(max(0.0, cash), d)
                    continue
                qty = buy_notional / px
                cash -= target
                costs["transaction_cost_eur"] += buy_fee
                if cash < -_ACCOUNTING_TOL:
                    raise AssertionError(f"NEGATIVE_CASH_AFTER_BUY:{cash}")
                if abs(cash) <= _ACCOUNTING_TOL:
                    cash = 0.0
                positions[action["ticker"]] = {
                    "qty": qty,
                    "entry_price": px,
                    "entry_urth": umap[d][0],
                    "entry_benchmark_index":benchmark_open_index[d],
                    "entry_date": d,
                    "entry_index": idx[d],
                    "peak_excess": 0.0,
                    "buy_notional": buy_notional,
                    "buy_fee": buy_fee,
                    "tax_basis": buy_notional + buy_fee,
                    "family_id": action.get("family_id"),
                    "generation_id": action.get("generation_id"),
                    "model_artifact_id": action.get("model_artifact_id"),
                    "entry_policy_id": action.get("entry_policy_id"),
                    "exit_policy_id": action.get("exit_policy_id"),
                    "exit_generation_id": action.get("exit_generation_id"),
                    "entry_score": action.get("score"),
                    "entry_score_rank": action.get("score_rank"),
                    "entry_threshold": action.get("entry_threshold"),
                    "entry_threshold_distance": action.get("entry_threshold_distance"),
                }
                entry_scores[action["ticker"]] = action.get("score", 0.0)

        current = by_date.get(d)
        valid = []
        if current is not None:
            tickers, scores, neg_scores, group_size = current
            limit = max(1, math.ceil(group_size * current_top_fraction))
            passing = int(np.searchsorted(neg_scores, -current_threshold, side="right")) if len(neg_scores) else 0
            take = min(limit, passing)
            valid = [{"ticker": tickers[i], "score": float(scores[i]),"score_rank":i+1,
                      "entry_threshold":float(current_threshold),
                      "entry_threshold_distance":float(scores[i])-float(current_threshold)} for i in range(take)]
        valid_tickers = {str(x["ticker"]) for x in valid}

        for t, pos in list(positions.items()):
            if (d, t) not in pmap:
                continue
            require_price_quality(d,t,"close")
            stock_close = pmap[(d, t)][1]
            urth_close = umap[d][1]
            entry_benchmark=float(pos.get("entry_benchmark_index",pos["entry_urth"]))
            excess = stock_close / pos["entry_price"] / (benchmark_close_index[d] / entry_benchmark) - 1
            pos["peak_excess"] = max(pos["peak_excess"], excess)
            held_close_sessions = idx[d] - pos["entry_index"] + 1
            exit_now = held_close_sessions >= policy.holding_days
            if policy.exit_family == "SIGNAL_DECAY":
                exit_now |= t not in valid_tickers
            elif policy.exit_family == "RELATIVE_STOP":
                exit_now |= excess <= policy.exit_value
            elif policy.exit_family == "TRAILING_RELATIVE_STOP":
                exit_now |= excess <= pos["peak_excess"] - policy.exit_value
            elif policy.exit_family == "TAKE_PROFIT_RELATIVE":
                exit_now |= excess >= policy.exit_value
            if exit_now and next_d is not None:
                pending[next_d].append({"kind": "sell", "ticker": t})

        chosen = [x for x in valid if str(x["ticker"]) not in positions]
        chosen = chosen[: policy.max_names]
        if len(positions) >= policy.max_names and policy.replacement == "REPLACE_WEAKEST" and chosen:
            weakest = min(positions, key=lambda t: entry_scores.get(t, -math.inf))
            strongest = float(chosen[0]["score"])
            if strongest > entry_scores.get(weakest, -math.inf) and next_d is not None:
                pending[next_d].append({"kind": "sell", "ticker": weakest})

        scheduled_sells = {a["ticker"] for a in pending.get(next_d, []) if a.get("kind") == "sell"} if next_d is not None else set()
        effective_positions = max(0, len(positions) - len(scheduled_sells))
        available = max(0, policy.max_names - effective_positions)
        chosen = chosen[:available]
        if next_d is not None and chosen:
            for row in chosen:
                lineage = active_generation or {"family_id": None, "generation_id": None, "entry_policy_id": policy.policy_id, "exit_policy_id": policy.exit_family}
                pending[next_d].append({"kind":"buy","ticker":str(row["ticker"]),
                    "score":float(row["score"]),"score_rank":int(row["score_rank"]),
                    "entry_threshold":float(row["entry_threshold"]),
                    "entry_threshold_distance":float(row["entry_threshold_distance"]),**lineage})

        total, cash_value, urth_value, stock_value, receivable_value = components(d, "close")
        accounting_error = total - (cash_value + urth_value + stock_value + receivable_value)
        if abs(accounting_error) > _ACCOUNTING_TOL * max(1.0, abs(total)):
            raise AssertionError(f"ACCOUNTING_IDENTITY_BROKEN:{d}:{accounting_error}")
        if total <= 0:
            raise AssertionError(f"NON_POSITIVE_NAV:{d}:{total}")
        stock_exposure = stock_value / total
        urth_exposure = urth_value / total
        cash_exposure = cash_value / total
        receivable_exposure = receivable_value / total
        if stock_exposure > policy.sleeve + 5e-6:
            sleeve_breach_count += 1
        if stock_exposure + urth_exposure + cash_exposure + receivable_exposure > 1.0 + 5e-6:
            raise AssertionError(f"EXPOSURE_SUM_EXCEEDED:{d}")
        if len(positions) > policy.max_names:
            raise AssertionError(f"MAX_NAMES_EXCEEDED:{d}:{len(positions)}>{policy.max_names}")
        curve.append(
            {
                "date": d,
                "strategy_value": total,
                "urth_value":initial*benchmark_close_index[d]/benchmark_initial_index_close,
                "distribution_cashflow_eur":strategy_distribution_cashflow,
                "benchmark_distribution_cashflow_eur":benchmark_cashflow,
                "positions": len(positions),
                "stock_exposure": stock_exposure,
                "urth_exposure": urth_exposure,
                "cash_exposure": cash_exposure,
                "distribution_receivable_value":receivable_value,
                "distribution_receivable_exposure":receivable_exposure,
                "accounting_error_eur": accounting_error,
                "regime": None,
            }
        )

    curve = pd.DataFrame(curve)
    curve = curve.reindex(columns=["date","strategy_value","urth_value","positions","stock_exposure",
                                   "urth_exposure","cash_exposure","accounting_error_eur",
                                   "distribution_receivable_value","distribution_receivable_exposure",
                                   "distribution_cashflow_eur","benchmark_distribution_cashflow_eur","regime"])
    if regime is not None:
        curve = curve.merge(regime[["date", "regime"]], on="date", how="left", suffixes=("", "_r"))
        curve["regime"] = curve["regime_r"].fillna(curve["regime"]).fillna("TRANSITION")
        curve = curve.drop(columns=["regime_r"])

    benchmark_after_tax, benchmark_tax = benchmark_tax_approximate(initial, float(curve["urth_value"].iloc[-1]), tax_config)
    metrics = _metrics(
        curve,
        initial,
        trades,
        costs,
        tax,
        initial,
        benchmark_terminal_after_tax=benchmark_after_tax if tax_config.enabled else None,
        benchmark_tax=benchmark_tax,
    )
    metrics["tax_world"] = "DE_RETAIL_TAX_AWARE" if tax_config.enabled else "PRE_TAX"
    metrics["sleeve_mark_to_market_breach_days"] = int(sleeve_breach_count)
    metrics["gross_distribution_income_eur"]=float(distribution_audit["gross_distribution_eur"])
    metrics["distribution_tax_paid_eur"]=float(distribution_audit["distribution_tax_eur"])
    metrics["distribution_event_count"]=int(distribution_audit["distribution_event_count"])
    metrics["benchmark_gross_distribution_income_eur"]=float(distribution_audit["benchmark_gross_distribution_eur"])
    metrics["benchmark_return_contract"]=("CASH_DISTRIBUTIONS_REINVESTED_AT_PAYABLE_DATE_OPEN"
                                          if distribution_ex_map else "PRICE_RETURN_ONLY_NO_DISTRIBUTION_INPUT")
    open_positions = [
        {
            "ticker": str(ticker),
            "entry_date": str(position["entry_date"]),
            "entry_price": float(position["entry_price"]),
            "qty": float(position["qty"]),
            "buy_notional": float(position["buy_notional"]),
            "buy_fee": float(position["buy_fee"]),
            "tax_basis": float(position["tax_basis"]),
            "family_id": position.get("family_id"),
            "generation_id": position.get("generation_id"),
            "entry_family_id": position.get("family_id"),
            "entry_generation_id": position.get("generation_id"),
            "entry_score": position.get("entry_score"),
            "entry_score_rank": position.get("entry_score_rank"),
            "entry_threshold": position.get("entry_threshold"),
            "entry_threshold_distance": position.get("entry_threshold_distance"),
            "entry_model_artifact_id": position.get("model_artifact_id"),
            "entry_policy_id": position.get("entry_policy_id"),
            "exit_policy_id": position.get("exit_policy_id"),
            "exit_generation_id": position.get("exit_generation_id"),
        }
        for ticker, position in sorted(positions.items())
    ]
    as_of=pd.Timestamp(curve["date"].iloc[-1])
    replay_state={
        "schema_version":"OPPORTUNITY_PORTFOLIO_REPLAY_STATE_V1","as_of":str(as_of),
        "session_cursor":int(idx[as_of]),"initial_value":float(initial),
        "benchmark_start_date":str(benchmark_start_date),"benchmark_initial_close":float(benchmark_initial_close),
        "cash":float(cash),"urth_units":float(urth_units),
        "positions":positions,"pending_orders":{str(key):value for key,value in pending.items()},
        "pending_distributions":{str(key):value for key,value in pending_distributions.items()},
        "distribution_audit":distribution_audit,
        "tax_ledger":tax.snapshot(),"transaction_cost_eur":float(costs["transaction_cost_eur"]),
        "entry_scores":entry_scores,"sleeve_breach_count":int(sleeve_breach_count),
        "curve":curve.to_dict(orient="records"),"trades":trades,"threshold":float(threshold),
    }
    return {
        "metrics": metrics,
        "curve": curve,
        "trades": trades,
        "open_positions": open_positions,
        "cost_audit": costs | {"roundtrip_bps": cost.roundtrip_bps, "per_side_bps": cost.per_side_bps},
        "tax_audit": tax.snapshot(),
        "distribution_audit":distribution_audit,
        "threshold": threshold,
        "replay_state": replay_state,
        "generation_schedule_applied": bool(lineage_schedule),
        "policy": asdict(policy),
    }
