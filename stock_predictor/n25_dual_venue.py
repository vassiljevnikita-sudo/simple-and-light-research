#!/usr/bin/env python3
"""Ex-ante dual-venue route selection for the frozen N25 research candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

BASE_MODEL_ID = "V4_N25_T212_RESEARCH_V1"
EXECUTION_POLICY_ID = "EXEC_N25_DUAL_VENUE_RESEARCH_V1"
PAPER_TRADING_ONLY = True
US_ROUTE = "US_PRIMARY_USD"
GERMAN_ROUTE = "GERMAN_EUR_SAME_ISIN"
NO_TRADE = "NO_TRADE_OR_DEFER"
VALID_FX_MODES = {"NONE", "EUR_AUTO_CONVERT_PER_ORDER", "PERSISTENT_USD_BALANCE"}


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field.upper()}_INVALID") from exc
    if not result.is_finite():
        raise ValueError(f"{field.upper()}_NONFINITE")
    return result


def _timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field.upper()}_INVALID") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field.upper()}_NAIVE")
    return parsed


@dataclass(frozen=True)
class RouteObservation:
    route_id: str
    isin: str
    currency: str
    decision_timestamp: str
    quote_timestamp: str
    bid: Decimal
    ask: Decimal
    expected_entry_slippage_bps: Decimal
    expected_exit_slippage_bps: Decimal
    explicit_fees_bps: Decimal
    expected_execution_delay_cost_bps: Decimal
    expected_nonfill_cost_bps: Decimal
    expected_basis_tracking_cost_bps: Decimal
    uncertainty_buffer_bps: Decimal
    quote_size_eur: Decimal
    order_notional_eur: Decimal
    max_quote_age_seconds: int
    fx_mode: str = "NONE"
    fx_fee_fraction_per_conversion: Decimal = Decimal("0.0015")
    allocated_fx_conversion_cost_bps: Decimal = Decimal("0")
    eligible: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RouteObservation":
        required = {
            "route_id", "isin", "currency", "decision_timestamp", "quote_timestamp",
            "bid", "ask", "expected_entry_slippage_bps", "expected_exit_slippage_bps",
            "explicit_fees_bps", "expected_execution_delay_cost_bps",
            "expected_nonfill_cost_bps", "expected_basis_tracking_cost_bps",
            "uncertainty_buffer_bps", "quote_size_eur", "order_notional_eur",
            "max_quote_age_seconds",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"ROUTE_FIELDS_MISSING:{','.join(missing)}")
        item = cls(
            route_id=str(raw["route_id"]),
            isin=str(raw["isin"]).upper(),
            currency=str(raw["currency"]).upper(),
            decision_timestamp=str(raw["decision_timestamp"]),
            quote_timestamp=str(raw["quote_timestamp"]),
            bid=_decimal(raw["bid"], "bid"),
            ask=_decimal(raw["ask"], "ask"),
            expected_entry_slippage_bps=_decimal(raw["expected_entry_slippage_bps"], "entry_slippage"),
            expected_exit_slippage_bps=_decimal(raw["expected_exit_slippage_bps"], "exit_slippage"),
            explicit_fees_bps=_decimal(raw["explicit_fees_bps"], "explicit_fees"),
            expected_execution_delay_cost_bps=_decimal(raw["expected_execution_delay_cost_bps"], "delay_cost"),
            expected_nonfill_cost_bps=_decimal(raw["expected_nonfill_cost_bps"], "nonfill_cost"),
            expected_basis_tracking_cost_bps=_decimal(raw["expected_basis_tracking_cost_bps"], "basis_cost"),
            uncertainty_buffer_bps=_decimal(raw["uncertainty_buffer_bps"], "uncertainty_buffer"),
            quote_size_eur=_decimal(raw["quote_size_eur"], "quote_size_eur"),
            order_notional_eur=_decimal(raw["order_notional_eur"], "order_notional_eur"),
            max_quote_age_seconds=int(raw["max_quote_age_seconds"]),
            fx_mode=str(raw.get("fx_mode", "NONE")),
            fx_fee_fraction_per_conversion=_decimal(
                raw.get("fx_fee_fraction_per_conversion", "0.0015"),
                "fx_fee_fraction_per_conversion",
            ),
            allocated_fx_conversion_cost_bps=_decimal(
                raw.get("allocated_fx_conversion_cost_bps", "0"),
                "allocated_fx_conversion_cost_bps",
            ),
            eligible=bool(raw.get("eligible", True)),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.route_id not in {US_ROUTE, GERMAN_ROUTE}:
            raise ValueError("ROUTE_ID_INVALID")
        if len(self.isin) != 12 or not self.isin.isalnum():
            raise ValueError("ISIN_INVALID")
        expected_currency = "USD" if self.route_id == US_ROUTE else "EUR"
        if self.currency != expected_currency:
            raise ValueError("ROUTE_CURRENCY_MISMATCH")
        if self.fx_mode not in VALID_FX_MODES:
            raise ValueError("FX_MODE_INVALID")
        if self.route_id == GERMAN_ROUTE and self.fx_mode != "NONE":
            raise ValueError("GERMAN_ROUTE_FX_MODE_INVALID")
        if self.route_id == US_ROUTE and self.fx_mode == "NONE":
            raise ValueError("US_ROUTE_FX_MODE_REQUIRED")
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("QUOTE_INVALID")
        if self.quote_size_eur < 0 or self.order_notional_eur <= 0:
            raise ValueError("NOTIONAL_INVALID")
        if self.max_quote_age_seconds < 0:
            raise ValueError("MAX_QUOTE_AGE_INVALID")
        if not Decimal("0") <= self.fx_fee_fraction_per_conversion < Decimal("1"):
            raise ValueError("FX_FEE_INVALID")
        for value in (
            self.expected_entry_slippage_bps,
            self.expected_exit_slippage_bps,
            self.explicit_fees_bps,
            self.expected_execution_delay_cost_bps,
            self.expected_nonfill_cost_bps,
            self.expected_basis_tracking_cost_bps,
            self.uncertainty_buffer_bps,
            self.allocated_fx_conversion_cost_bps,
        ):
            if value < 0:
                raise ValueError("NEGATIVE_COST_COMPONENT")
        decision = _timestamp(self.decision_timestamp, "decision_timestamp")
        quote = _timestamp(self.quote_timestamp, "quote_timestamp")
        if quote > decision:
            raise ValueError("LOOKAHEAD_QUOTE")
        if (decision - quote).total_seconds() > self.max_quote_age_seconds:
            raise ValueError("QUOTE_STALE")

    @property
    def spread_bps(self) -> Decimal:
        mid = (self.bid + self.ask) / Decimal("2")
        return (self.ask - self.bid) / mid * Decimal("10000")

    @property
    def fx_cost_bps(self) -> Decimal:
        if self.route_id != US_ROUTE:
            return Decimal("0")
        if self.fx_mode == "EUR_AUTO_CONVERT_PER_ORDER":
            return self.fx_fee_fraction_per_conversion * Decimal("2") * Decimal("10000")
        if self.fx_mode == "PERSISTENT_USD_BALANCE":
            return self.allocated_fx_conversion_cost_bps
        raise ValueError("US_ROUTE_FX_MODE_REQUIRED")

    @property
    def expected_all_in_cost_bps(self) -> Decimal:
        size_shortfall = max(Decimal("0"), self.order_notional_eur - self.quote_size_eur)
        size_penalty = (
            Decimal("0")
            if self.order_notional_eur == 0
            else size_shortfall / self.order_notional_eur * Decimal("10000")
        )
        return (
            self.spread_bps
            + self.expected_entry_slippage_bps
            + self.expected_exit_slippage_bps
            + self.explicit_fees_bps
            + self.fx_cost_bps
            + self.expected_execution_delay_cost_bps
            + self.expected_nonfill_cost_bps
            + self.expected_basis_tracking_cost_bps
            + self.uncertainty_buffer_bps
            + size_penalty
        )


@dataclass(frozen=True)
class RouteDecision:
    base_model_id: str
    execution_policy_id: str
    decision_timestamp: str
    isin: str
    selected_route_id: str
    expected_edge_bps: Decimal
    expected_all_in_cost_bps: Decimal | None
    expected_remaining_edge_bps: Decimal | None
    reason: str
    paper_trading_only: bool
    route_costs_bps: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["expected_edge_bps"] = str(self.expected_edge_bps)
        data["expected_all_in_cost_bps"] = (
            None if self.expected_all_in_cost_bps is None else str(self.expected_all_in_cost_bps)
        )
        data["expected_remaining_edge_bps"] = (
            None if self.expected_remaining_edge_bps is None else str(self.expected_remaining_edge_bps)
        )
        return data


def choose_route(
    observations: Iterable[RouteObservation],
    *,
    expected_edge_bps: Any,
    minimum_remaining_edge_bps: Any = "0",
) -> RouteDecision:
    routes = list(observations)
    if not routes:
        raise ValueError("NO_ROUTE_OBSERVATIONS")
    edge = _decimal(expected_edge_bps, "expected_edge_bps")
    hurdle = _decimal(minimum_remaining_edge_bps, "minimum_remaining_edge_bps")
    if hurdle < 0:
        raise ValueError("MINIMUM_REMAINING_EDGE_NEGATIVE")
    first = routes[0]
    if any(item.isin != first.isin for item in routes):
        raise ValueError("ROUTE_ISIN_MISMATCH")
    if any(item.decision_timestamp != first.decision_timestamp for item in routes):
        raise ValueError("ROUTE_DECISION_TIMESTAMP_MISMATCH")
    if len({item.route_id for item in routes}) != len(routes):
        raise ValueError("DUPLICATE_ROUTE")
    route_costs = {
        item.route_id: str(item.expected_all_in_cost_bps)
        for item in routes
        if item.eligible
    }
    eligible = [item for item in routes if item.eligible]
    if not eligible:
        return RouteDecision(
            BASE_MODEL_ID, EXECUTION_POLICY_ID, first.decision_timestamp, first.isin,
            NO_TRADE, edge, None, None, "NO_ELIGIBLE_ROUTE", PAPER_TRADING_ONLY, route_costs,
        )
    selected = min(eligible, key=lambda item: (item.expected_all_in_cost_bps, item.route_id))
    remaining = edge - selected.expected_all_in_cost_bps
    if remaining <= hurdle:
        return RouteDecision(
            BASE_MODEL_ID, EXECUTION_POLICY_ID, first.decision_timestamp, first.isin,
            NO_TRADE, edge, selected.expected_all_in_cost_bps, remaining,
            "EXPECTED_REMAINING_EDGE_NOT_ABOVE_HURDLE", PAPER_TRADING_ONLY, route_costs,
        )
    return RouteDecision(
        BASE_MODEL_ID, EXECUTION_POLICY_ID, first.decision_timestamp, first.isin,
        selected.route_id, edge, selected.expected_all_in_cost_bps, remaining,
        "MINIMUM_EXPECTED_ALL_IN_COST_EX_ANTE", PAPER_TRADING_ONLY, route_costs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    observations = [RouteObservation.from_mapping(item) for item in payload["routes"]]
    decision = choose_route(
        observations,
        expected_edge_bps=payload["expected_edge_bps"],
        minimum_remaining_edge_bps=payload.get("minimum_remaining_edge_bps", "0"),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(decision.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
