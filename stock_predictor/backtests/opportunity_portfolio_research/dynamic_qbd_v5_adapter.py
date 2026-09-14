"""Dynamic-QBD adapter for the canonical V5 H1-H30 research primitives.

The legacy ``stock_predictor.v5.training_entrypoint`` is intentionally not
used here because its contract is H5/H10/H20 walk-forward evaluation, not a
monthly production generation.  The compatibility names below route callers
to the production-fit H1-H30 implementation.
"""
from __future__ import annotations

from typing import Protocol
from pathlib import Path

from .dynamic_qbd_h1_30_adapter import (
    H130ProductionGenerationBuilder,
    ParquetH130DatasetMaterializer,
)


class CausalDatasetMaterializer(Protocol):
    def materialize(self, *, family, information_cutoff, latest_matured_label_cutoff, output_root: Path) -> dict: ...


class V5GenerationBuilder(H130ProductionGenerationBuilder):
    """Backward-compatible name for the authoritative H1-H30 builder."""


__all__ = ["CausalDatasetMaterializer", "ParquetH130DatasetMaterializer", "V5GenerationBuilder"]
