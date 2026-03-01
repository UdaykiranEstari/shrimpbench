"""
Generate assets/demo.svg — a pixel-perfect SVG of the ShrimpBench dashboard.

Shows two back-to-back runs:
  1. Passing case  — green footer, valid=True
  2. Failing case  — red footer, valid=False (mismatch detected)

Usage:
    python3 scripts/make_demo.py

Re-run this whenever the dashboard UI changes.
"""

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shrimpbench import run_benchmark

recording = Console(
    record=True,
    force_terminal=True,
    width=100,
    color_system="truecolor",
)

# ── Sample data ────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "category": ["Electronics", "Books", "Clothing", "Electronics", "Books"],
    "sales": [1500.0, 800.0, 1200.0, 2100.0, 950.0],
    "units": [30, 80, 60, 42, 95],
})

# ── Case 1: PASS ───────────────────────────────────────────────────────────────
def baseline(df):
    return df.groupby("category").agg({"sales": "sum", "units": "sum"}).reset_index()

def optimized(df):
    return df.groupby("category", sort=False).agg({"sales": "sum", "units": "sum"}).reset_index()

run_benchmark(
    baseline,
    optimized,
    scope={"df": df},
    alias_old="Baseline",
    alias_new="Optimized",
    user_name="Uday",
    merge_keys={0: ["category"]},
    show_output_heads=True,
    iterations=1,
    warmup_runs=1,
    output_dir=None,
    console=recording,
)

# ── Case 2: FAIL ───────────────────────────────────────────────────────────────
def buggy_optimized(df):
    result = df.groupby("category", sort=False).agg({"sales": "sum", "units": "sum"}).reset_index()
    result.loc[0, "sales"] = result.loc[0, "sales"] * 1.5  # intentional bug
    return result

run_benchmark(
    baseline,
    buggy_optimized,
    scope={"df": df},
    alias_old="Baseline",
    alias_new="Buggy Optimized",
    user_name="Uday",
    merge_keys={0: ["category"]},
    show_output_heads=True,
    iterations=1,
    warmup_runs=0,
    output_dir=None,
    console=recording,
)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent.parent / "assets" / "demo.svg"
recording.save_svg(str(out_path), title="ShrimpBench")
print(f"Saved: {out_path}")
