from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from .tax_contracts import TaxConfig

@dataclass
class TaxLedger:
    config: TaxConfig
    loss_carryforward: float = 0.0
    allowance_used: float = 0.0
    current_year: int | None = None
    tax_paid: float = 0.0
    realized_gain: float = 0.0
    realized_loss: float = 0.0
    used_allowance: float = 0.0
    distribution_income: float = 0.0
    distribution_tax_paid: float = 0.0

    def _year_reset(self, year: int) -> None:
        if self.current_year != year:
            self.current_year = year; self.allowance_used = 0.0

    def realize_stock_trade(self, sale_date: date, proceeds: float, basis: float, costs: float,
                            taxable_fraction: float = 1.0) -> float:
        self._year_reset(sale_date.year)
        gain = (proceeds - basis - costs) * float(taxable_fraction)
        if gain < 0:
            self.realized_loss += -gain; self.loss_carryforward += -gain; return 0.0
        self.realized_gain += gain
        offset = min(self.loss_carryforward, gain); self.loss_carryforward -= offset; taxable = gain - offset
        allowance = max(0.0, self.config.allowance_eur - self.allowance_used)
        used = min(allowance, taxable); self.allowance_used += used; self.used_allowance += used; taxable -= used
        if not self.config.enabled or taxable <= 0: return 0.0
        rate = self.config.capital_gains_rate * (1.0 + self.config.solidarity_surcharge + self.config.church_tax_rate)
        tax = max(0.0, taxable * rate); self.tax_paid += tax; return tax

    def realize_cash_distribution(self, payment_date: date, gross_income: float,
                                  taxable_fraction: float = 1.0) -> float:
        """Tax cash income without using the stock-loss carryforward."""
        self._year_reset(payment_date.year)
        gross=max(0.0,float(gross_income)); self.distribution_income += gross
        taxable=gross*float(taxable_fraction)
        allowance=max(0.0,self.config.allowance_eur-self.allowance_used)
        used=min(allowance,taxable); self.allowance_used += used; self.used_allowance += used; taxable -= used
        if not self.config.enabled or taxable <= 0: return 0.0
        rate=self.config.capital_gains_rate*(1.0+self.config.solidarity_surcharge+self.config.church_tax_rate)
        tax=max(0.0,taxable*rate); self.tax_paid += tax; self.distribution_tax_paid += tax; return tax

    def snapshot(self) -> dict:
        return {"realized_gain": self.realized_gain, "realized_loss": self.realized_loss, "tax_paid": self.tax_paid,
                "distribution_income":self.distribution_income,"distribution_tax_paid":self.distribution_tax_paid,
                "used_sparer_pauschbetrag": self.used_allowance, "remaining_loss_carryforward": self.loss_carryforward,
                "allowance_used_current_year": self.allowance_used, "current_year": self.current_year}

    @classmethod
    def from_snapshot(cls, config: TaxConfig, snapshot: dict | None) -> "TaxLedger":
        snapshot = snapshot or {}
        return cls(
            config=config,
            loss_carryforward=float(snapshot.get("remaining_loss_carryforward", 0.0)),
            allowance_used=float(snapshot.get("allowance_used_current_year", 0.0)),
            current_year=(int(snapshot["current_year"]) if snapshot.get("current_year") is not None else None),
            tax_paid=float(snapshot.get("tax_paid", 0.0)),
            realized_gain=float(snapshot.get("realized_gain", 0.0)),
            realized_loss=float(snapshot.get("realized_loss", 0.0)),
            used_allowance=float(snapshot.get("used_sparer_pauschbetrag", 0.0)),
            distribution_income=float(snapshot.get("distribution_income",0.0)),
            distribution_tax_paid=float(snapshot.get("distribution_tax_paid",0.0)),
        )

def benchmark_tax_approximate(initial: float, terminal: float, config: TaxConfig) -> tuple[float, float]:
    if not config.enabled: return terminal, 0.0
    gain = max(0.0, terminal - initial - config.allowance_eur)
    tax = gain * config.capital_gains_rate * (1.0 + config.solidarity_surcharge + config.church_tax_rate)
    return terminal - tax, tax
