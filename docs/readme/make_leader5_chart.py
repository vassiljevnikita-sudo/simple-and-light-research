"""Draw the README Leader5 chart from committed result files.

Reads only these files (no market data, no network):

- artifacts/leader5-selector-development-20261011/leader-sample.json
- artifacts/leader5-selector-development-20261011/leader-oos.json

Writes docs/readme/leader5-chart.png and docs/readme/leader5-chart-dark.png.

Usage, from the repository root:

    pip install matplotlib
    python docs/readme/make_leader5_chart.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
LEADER5 = ROOT / "artifacts/leader5-selector-development-20261011"

# Same surfaces and ink as make_results_chart.py; URTH is a neutral dashed reference line.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "line": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "line": "#3987e5"},
}

PANELS = (
    ("leader-sample.json", "Sample 2017-2021\nrule development"),
    ("leader-oos.json", "Validation 2022-2025\nalso used to choose the rule, not independent"),
)


def load(name: str) -> tuple[list[date], list[float], list[float]]:
    points = json.loads((LEADER5 / name).read_text(encoding="utf-8"))["points"]
    days = [date.fromisoformat(p["date"]) for p in points]
    return days, [p["navEUR"] for p in points], [p["benchmarkNavEUR"] for p in points]


def draw(mode: str) -> Path:
    theme = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150, sharey=True,
                             gridspec_kw={"wspace": 0.42})
    fig.patch.set_facecolor(theme["surface"])
    for ax, (name, title) in zip(axes, PANELS):
        days, strategy, urth = load(name)
        ax.plot(days, urth, color=theme["muted"], linewidth=1.4, linestyle=(0, (4, 3)))
        ax.plot(days, strategy, color=theme["line"], linewidth=2, solid_capstyle="round",
                solid_joinstyle="round")
        ax.plot([days[-1]], [strategy[-1]], "o", markersize=5, color=theme["line"])
        ax.text(days[-1], strategy[-1], f"  Leader5\n  EUR {strategy[-1]:,.0f}", va="center",
                ha="left", fontsize=9, color=theme["ink"])
        ax.text(days[-1], urth[-1], f"  URTH\n  EUR {urth[-1]:,.0f}", va="center", ha="left",
                fontsize=9, color=theme["ink2"])
        ax.set_facecolor(theme["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["axis"])
        ax.tick_params(colors=theme["ink2"], labelsize=9, length=0)
        ax.yaxis.grid(True, color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}k"))
        ax.set_title(title, loc="left", fontsize=10.5, color=theme["ink"], y=1.03, pad=0)
    axes[0].set_ylabel("EUR, start 10,000", fontsize=9, color=theme["ink2"])
    fig.text(0.01, 0.02,
             "Backtest, price-only, 0.25% cost per stock trade side. Research only, no real money. "
             "Source: artifacts/leader5-selector-development-20261011; "
             "script: docs/readme/make_leader5_chart.py",
             fontsize=7.5, color=theme["muted"])
    fig.subplots_adjust(left=0.07, right=0.88, top=0.84, bottom=0.14)
    suffix = "" if mode == "light" else "-dark"
    path = OUT_DIR / f"leader5-chart{suffix}.png"
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for theme_name in THEMES:
        print(draw(theme_name).relative_to(ROOT))
