#!/usr/bin/env python3
"""Benchmark cost policy for an investable MSCI World ETF implementation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

BENCHMARK_POLICY_ID = "MSCI_WORLD_IMPLEMENTABLE_PROXY_V1"
DEFAULT_ANNUAL_TER = Decimal("0.002")
DAYS_PER_YEAR = Decimal("365.2425")


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field.upper()}_INVALID") from exc
    if not result.is_finite():
        raise ValueError(f"{field.upper()}_NONFINITE")
    return result


@dataclass(frozen=True)
class BenchmarkSource:
    source_id: str
    source_kind: str
    expenses_embedded: bool
    annual_ter_fraction: Decimal = DEFAULT_ANNUAL_TER

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("BENCHMARK_SOURCE_ID_MISSING")
        if self.source_kind not in {"INDEX_OR_GROSS_PROXY", "ETF_MARKET_PRICE", "ETF_NAV"}:
            raise ValueError("BENCHMARK_SOURCE_KIND_INVALID")
        if not Decimal("0") <= self.annual_ter_fraction < Decimal("1"):
            raise ValueError("BENCHMARK_TER_INVALID")
        if self.source_kind in {"ETF_MARKET_PRICE", "ETF_NAV"} and not self.expenses_embedded:
            raise ValueError("ETF_EXPENSE_EMBEDDED_STATUS_CONTRADICTORY")

    def audit_metadata(self, elapsed_days: Decimal) -> dict[str, object]:
        self.validate()
        return {
            "benchmark_policy_id": BENCHMARK_POLICY_ID,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "expenses_embedded": self.expenses_embedded,
            "annual_ter_fraction": str(self.annual_ter_fraction),
            "elapsed_days": str(elapsed_days),
            "ter_applied_separately": not self.expenses_embedded,
            "double_counting_forbidden": True,
        }


def ter_growth_multiplier(
    *,
    elapsed_days: Any,
    annual_ter_fraction: Any = DEFAULT_ANNUAL_TER,
) -> Decimal:
    """Return the geometric holding-period multiplier after annual TER."""
    days = _decimal(elapsed_days, "elapsed_days")
    ter = _decimal(annual_ter_fraction, "annual_ter_fraction")
    if days < 0:
        raise ValueError("ELAPSED_DAYS_NEGATIVE")
    if not Decimal("0") <= ter < Decimal("1"):
        raise ValueError("BENCHMARK_TER_INVALID")
    if days == 0 or ter == 0:
        return Decimal("1")
    with localcontext() as context:
        context.prec = 34
        exponent = days / DAYS_PER_YEAR
        return (Decimal("1") - ter) ** exponent


def apply_benchmark_policy(
    *,
    gross_return_fraction: Any,
    elapsed_days: Any,
    source: BenchmarkSource,
) -> dict[str, object]:
    """Apply TER exactly once and return an auditable result."""
    gross_return = _decimal(gross_return_fraction, "gross_return_fraction")
    days = _decimal(elapsed_days, "elapsed_days")
    source.validate()
    gross_growth = Decimal("1") + gross_return
    if gross_growth < 0:
        raise ValueError("BENCHMARK_GROWTH_NEGATIVE")
    multiplier = (
        Decimal("1")
        if source.expenses_embedded
        else ter_growth_multiplier(
            elapsed_days=days,
            annual_ter_fraction=source.annual_ter_fraction,
        )
    )
    net_growth = gross_growth * multiplier
    return {
        **source.audit_metadata(days),
        "gross_return_fraction": str(gross_return),
        "ter_growth_multiplier": str(multiplier),
        "net_return_fraction": str(net_growth - Decimal("1")),
    }


__all__ = [
    "BENCHMARK_POLICY_ID",
    "BenchmarkSource",
    "DEFAULT_ANNUAL_TER",
    "apply_benchmark_policy",
    "ter_growth_multiplier",
]
