"""Check the README chart and links against the committed files.

This is an artifact check, not a reproduction of the experiments. It does
three things, from the repository root:

1. Redraws the README chart with make_results_chart.py, which reads only the
   committed summary.json files listed in that script.
2. Compares the redrawn PNGs pixel by pixel with the committed PNGs. PNG
   metadata is ignored. Any changed pixel fails the check.
3. Checks that every relative link and image path in README.md points to a
   file or directory that exists in the repository.

Usage:

    pip install matplotlib==3.11.2
    python docs/readme/verify_readme_artifacts.py

Exit code 0 when all checks pass, 1 otherwise. The committed PNGs are
overwritten by the redraw; run `git diff` afterwards to see any change.
"""

from __future__ import annotations

import io
import re
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT / "docs/readme"
CHART_SCRIPT = CHART_DIR / "make_results_chart.py"
PNGS = ("results-chart.png", "results-chart-dark.png")
README = ROOT / "README.md"

LINK_PATTERNS = (
    re.compile(r"\]\(([^)\s]+)\)"),  # markdown links and images
    re.compile(r'\b(?:src|srcset|href)="([^"]+)"'),  # HTML attributes
)


def compare_charts() -> list[str]:
    committed = {name: (CHART_DIR / name).read_bytes() for name in PNGS}
    runpy.run_path(str(CHART_SCRIPT), run_name="__main__")
    errors = []
    for name in PNGS:
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


def main() -> int:
    errors = compare_charts() + check_links()
    for error in errors:
        print(f"FAIL {error}")
    if errors:
        return 1
    print("PASS chart matches the committed summary.json files; README links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
