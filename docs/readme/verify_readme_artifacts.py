"""Check the README chart and links against the committed files.

This is an artifact check, not a reproduction of the experiments. It does
four things, from the repository root:

1. Redraws the README charts with make_results_chart.py and
   make_leader5_chart.py, which read only the committed result files listed in
   those scripts.
2. Compares the redrawn PNGs pixel by pixel with the committed PNGs. PNG
   metadata is ignored. Any changed pixel fails the check.
3. Checks that every relative link and image path in README.md points to a
   file or directory that exists in the repository.
4. Checks the Leader5 numbers: the end values in the Leader5 summary.json
   equal the last points of its committed curves, and the README quotes the
   summary's end values and mean yearly excess.

Usage:

    pip install matplotlib==3.11.2
    python docs/readme/verify_readme_artifacts.py

Exit code 0 when all checks pass, 1 otherwise. The committed PNGs are
overwritten by the redraw; run `git diff` afterwards to see any change.
"""

from __future__ import annotations

import io
import json
import re
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT / "docs/readme"
CHARTS = {
    "make_results_chart.py": ("results-chart.png", "results-chart-dark.png"),
    "make_leader5_chart.py": ("leader5-chart.png", "leader5-chart-dark.png"),
}
README = ROOT / "README.md"
LEADER5 = ROOT / "artifacts/leader5-selector-development-20261011"

LINK_PATTERNS = (
    re.compile(r"\]\(([^)\s]+)\)"),  # markdown links and images
    re.compile(r'\b(?:src|srcset|href)="([^"]+)"'),  # HTML attributes
)


def compare_charts() -> list[str]:
    errors = []
    for script, pngs in CHARTS.items():
        errors += compare_chart(CHART_DIR / script, pngs)
    return errors


def compare_chart(script: Path, pngs: tuple[str, ...]) -> list[str]:
    committed = {name: (CHART_DIR / name).read_bytes() for name in pngs}
    runpy.run_path(str(script), run_name="__main__")
    errors = []
    for name in pngs:
        new_bytes = (CHART_DIR / name).read_bytes()
        if new_bytes == committed[name]:
            print(f"chart {name}: byte-identical to the committed file")
            continue
        old = mpimg.imread(io.BytesIO(committed[name]), format="png")
        new = mpimg.imread(io.BytesIO(new_bytes), format="png")
        if old.shape != new.shape:
            errors.append(f"chart {name}: size changed from {old.shape} to {new.shape}")
            continue
        changed = int((abs(old - new) > 0).any(axis=-1).sum())
        if changed:
            errors.append(f"chart {name}: {changed} pixels differ from the committed file")
        else:
            print(f"chart {name}: same pixels, only PNG metadata differs")
    return errors


def check_links() -> list[str]:
    text = README.read_text(encoding="utf-8")
    targets = sorted({m for pattern in LINK_PATTERNS for m in pattern.findall(text)})
    errors = []
    checked = 0
    for target in targets:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue  # external URL, mailto or in-page anchor
        path = target.split("#", 1)[0].split("?", 1)[0]
        checked += 1
        if not (ROOT / path).exists():
            errors.append(f"README link target missing: {target}")
    print(f"README: {checked} relative link targets checked")
    return errors


def check_leader5() -> list[str]:
    summary = json.loads((LEADER5 / "summary.json").read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    errors = []
    for key, period in summary["periods"].items():
        last = json.loads((LEADER5 / period["file"]).read_text(encoding="utf-8"))["points"][-1]
        for field, curve_field in (("terminalEUR", "navEUR"), ("urthTerminalEUR", "benchmarkNavEUR")):
            if round(last[curve_field], 2) != period[field]:
                errors.append(f"Leader5 {key}: {field} {period[field]} differs from {period['file']}")
            quoted = f"EUR {period[field]:,.0f}"
            if quoted not in readme:
                errors.append(f"Leader5 {key}: README does not quote {field} as {quoted}")
        mean = period.get("meanExcessEURPerYear")
        if mean is not None and f"EUR {mean:,.0f}" not in readme:
            errors.append(f"Leader5 {key}: README does not quote the mean yearly excess EUR {mean:,.0f}")
    print(f"Leader5: {len(summary['periods'])} periods checked against curves and README")
    return errors


def main() -> int:
    errors = compare_charts() + check_links() + check_leader5()
    for error in errors:
        print(f"FAIL {error}")
    if errors:
        return 1
    print("PASS charts match the committed result files; README links resolve; Leader5 numbers match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
