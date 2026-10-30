"""Draw the README results chart from committed result files.

Reads only these files (no market data, no network):

- artifacts/dynamic-qbd-fold-clock-refit-h3-d2-n1-rerun2/summary.json
- artifacts/dynamic-qbd-abc-recalibration-h3-d2-n1/summary.json
- artifacts/dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826/summary.json

Writes docs/readme/results-chart.png and docs/readme/results-chart-dark.png.

Usage, from the repository root:

    pip install matplotlib
    python docs/readme/make_results_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

FOLD_CLOCK = ROOT / "artifacts/dynamic-qbd-fold-clock-refit-h3-d2-n1-rerun2/summary.json"
ABC = ROOT / "artifacts/dynamic-qbd-abc-recalibration-h3-d2-n1/summary.json"
SURFACE = ROOT / "artifacts/dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826/summary.json"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "bar": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "bar": "#3987e5",
    },
}

ARM_LABELS = {
    "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS": "Retrain only when a new block\nof outcomes is complete",
    "A_FROZEN_MODEL_FROZEN_CALIBRATION": "Train once, never retrain",
    "M_MONTHLY_REFIT_FROZEN_RECIPE": "Retrain every month",
    "B_FROZEN_MODEL_ROLLING_RECALIBRATION": "Train once, re-tune the buy\nthreshold every month",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def single_cell_rows() -> tuple[list[tuple[str, float]], float, dict]:
    """Terminal value of EUR 10,000 per arm for the H3/D2/N1 setting."""
    fold = load(FOLD_CLOCK)
    abc = load(ABC)
    rows: dict[str, dict] = {}
    for summary in (fold, abc):
        for row in summary["portfolio_value_comparison"]:
            rows.setdefault(row["strategy"], row)
    rows.pop("C_MONTHLY_REFIT_FROZEN_RECIPE", None)  # same run as M, listed twice
    values = [(ARM_LABELS[k], float(v["terminal_value"])) for k, v in rows.items()]
    values.sort(key=lambda item: item[1])
    benchmarks = {round(float(v["urth_terminal_value"]), 2) for v in rows.values()}
    if len(benchmarks) != 1:
        raise SystemExit(f"benchmark values differ between runs: {benchmarks}")
    return values, benchmarks.pop(), fold["experiment"]


def surface_rows() -> tuple[list[tuple[str, float, str]], int]:
    """Median yearly benchmark-excess change over all FIXED families."""
    gates = load(SURFACE)["gates"]
    f_a = gates["F_MINUS_A"]["FIXED"]
    m_f = gates["M_MINUS_F"]["FIXED"]
    years = int(f_a["year_count"])
    rows = [
        (
            "Retrain every month\nvs. retrain on new outcomes",
            100 * float(m_f["median_family_cagr_excess_delta"]),
            f"worse in {round(m_f['directional_year_fraction'] * years)} of {years} years "
            f"and {m_f['directional_plateau_fraction']:.0%} of model groups",
        ),
        (
            "Retrain on new outcomes\nvs. never retrain",
            100 * float(f_a["median_family_cagr_excess_delta"]),
            f"better in only {round(f_a['directional_year_fraction'] * years)} of {years} years;\n"
            f"largest fall vs. benchmark {abs(100 * f_a['median_plateau_relative_maxdrawdown_delta']):.0f} pp deeper",
        ),
    ]
    return rows, int(f_a["family_count"])


def style_axis(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["ink2"], labelsize=9, length=0)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for label in ax.get_yticklabels():
        label.set_color(theme["ink"])
        label.set_fontsize(9.5)


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    cell, benchmark, experiment = single_cell_rows()
    surface, family_count = surface_rows()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75}
    )
    fig.patch.set_facecolor(theme["surface"])

    # Left panel: one setting, money at the end.
    labels = [name for name, _ in cell]
    values = [value for _, value in cell]
    bars = left.barh(labels, values, height=0.56, color=theme["bar"])
    for bar, value in zip(bars, values):
        left.text(value + 250, bar.get_y() + bar.get_height() / 2, f"EUR {value:,.0f}",
                  va="center", ha="left", fontsize=9, color=theme["ink"])
    left.axvline(benchmark, color=theme["muted"], linestyle=(0, (4, 3)), linewidth=1.2)
    left.text(benchmark - 250, len(values) - 0.45, f"index fund (URTH): EUR {benchmark:,.0f}",
              ha="right", va="center", fontsize=8.5, color=theme["ink2"])
    left.set_ylim(-0.6, len(values) - 0.15)
    left.set_xlim(0, max(values) * 1.28)
    left.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5000))
    left.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    style_axis(left, theme)
    left.set_title(
        f"One setting, EUR 10,000 invested\n{experiment['evaluation_start']} to {experiment['evaluation_end']}",
        loc="left", fontsize=10.5, color=theme["ink"], y=1.03, pad=0,
    )

    # Right panel: all FIXED families (fixed holding period), median change in yearly return above benchmark.
    labels = [name for name, _, _ in surface]
    values = [value for _, value, _ in surface]
    notes = [note for _, _, note in surface]
    bars = right.barh(labels, values, height=0.5, color=theme["bar"])
    right.axvline(0, color=theme["axis"], linewidth=1.2)
    for bar, value, note in zip(bars, values, notes):
        y = bar.get_y() + bar.get_height() / 2
        offset = 0.15 if value >= 0 else -0.15
        right.text(value + offset, y, f"{value:+.1f} pp", va="center",
                   ha="left" if value >= 0 else "right", fontsize=9, color=theme["ink"])
        right.text(-6.9, y - 0.45, note, va="center", ha="left", fontsize=8, color=theme["ink2"])
    right.set_xlim(-7, 3)
    right.set_ylim(-0.75, len(values) - 0.4)
    right.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}" if x else "0"))
    style_axis(right, theme)
    right.set_title(
        f"{family_count:,} fixed-holding families, 2016-2025\n"
        "median change in yearly return above benchmark",
        loc="left", fontsize=10.5, color=theme["ink"], y=1.03, pad=0,
    )
    right.set_xlabel("percentage points (pp)", fontsize=9, color=theme["ink2"])

    fig.text(0.01, 0.02,
             "Research only. Development data, no real money. Source: artifacts/*/summary.json; "
             "script: docs/readme/make_results_chart.py",
             fontsize=7.5, color=theme["muted"])
    fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.17)

    suffix = "" if mode == "light" else "-dark"
    path = OUT_DIR / f"results-chart{suffix}.png"
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for theme_name in THEMES:
        print(draw(theme_name).relative_to(ROOT))
