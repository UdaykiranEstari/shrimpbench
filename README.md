# ShrimpBench

**Zero-effort benchmarking for Python functions with smart DataFrame validation and a Rich CLI dashboard.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/shrimpbench)](https://pypi.org/project/shrimpbench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Why ShrimpBench?

Comparing two implementations of the same function shouldn't require boilerplate for timing, memory tracking, output validation, and reporting. ShrimpBench wraps all of that into a single function call:

- Pass in your **baseline** and **optimized** functions
- ShrimpBench auto-discovers parameters, runs both, and gives you a full performance report
- Output DataFrames are automatically validated row-by-row with configurable tolerance

No setup. No config files. One function call.

## Installation

```bash
pip install shrimpbench
```

## Quick Start

```python
from shrimpbench import run_benchmark

def baseline(df):
    return df.groupby("category").sum()

def optimized(df):
    return df.groupby("category", sort=False).sum()

result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope={"df": my_dataframe},
    alias_old="Pandas GroupBy",
    alias_new="Pandas GroupBy (no sort)",
)

print(f"Speedup: {result.speedup:.2f}x")
print(f"Valid: {result.valid}")
```

## Features

### Automatic Parameter Discovery
ShrimpBench inspects function signatures and maps variables from the `scope` dict automatically. No need to manually wire up arguments.

```python
# Both functions receive only the parameters they need
def baseline(df, threshold):
    ...

def optimized(df, threshold, use_cache=True):
    ...

run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope={"df": data, "threshold": 0.5, "use_cache": True},
)
```

### Execution Time and Memory Measurement
Each function is timed with `time.perf_counter` and memory-tracked with `tracemalloc`. Results are displayed in a formatted table with deltas and percentage changes.

### Multi-Iteration Mode
Run multiple iterations to get statistically stable results. Reports min, mean, median, and standard deviation.

```python
result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope=inputs,
    iterations=5,
    warmup_runs=2,
)
```

### Smart DataFrame Validation
When functions return DataFrames (or tuples of DataFrames), ShrimpBench automatically:
- Aligns rows by sort keys or merge keys
- Compares numeric columns within tolerance (`rtol`, `atol`)
- Reports column mismatches, row count differences, and value-level discrepancies
- Exports mismatch details to CSV for debugging

```python
result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope=inputs,
    merge_keys=["id", "date"],  # Align rows by these columns
    rtol=0.01,                  # 1% relative tolerance
    atol=1e-5,                  # Absolute tolerance
)
```

### Multi-DataFrame Tuple Support
If your functions return multiple DataFrames as a tuple, ShrimpBench validates each one independently. Use per-index merge keys for precise alignment:

```python
result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope=inputs,
    merge_keys={
        0: ["item_id", "location"],   # Keys for first DataFrame
        1: ["date", "category"],       # Keys for second DataFrame
    },
)
```

### Line Profiler Integration
Attach a line profiler to specific functions to identify bottlenecks. Results are saved to a text file.

```python
from my_module import slow_helper

result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope=inputs,
    profile_funcs=[optimized, slow_helper],
    output_dir="benchmark_outputs/",
)
# Profiler output saved to benchmark_outputs/profile_results.txt
```

### Historical Tracking
Append results to a JSON file to track performance over time.

```python
result = run_benchmark(
    func_old=baseline,
    func_new=optimized,
    scope=inputs,
    history_file="benchmark_history.json",
    output_dir="benchmark_outputs/",
)
```

### Rich CLI Dashboard
ShrimpBench renders a full terminal dashboard using [Rich](https://github.com/Textualize/rich), including:
- ASCII art header with run metadata
- Side-by-side DataFrame previews (first 5 sorted rows)
- Color-coded results table with speedup, time, and memory
- Pass/fail validation summary panel

## API Reference

### `run_benchmark(...) -> BenchmarkResult`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `func_old` | `callable` | *required* | Baseline function |
| `func_new` | `callable` | *required* | Optimized function |
| `scope` | `dict` | *required* | Variables to auto-map as function arguments |
| `alias_old` | `str` | `"Baseline"` | Display name for baseline |
| `alias_new` | `str` | `"Optimized"` | Display name for optimized |
| `merge_keys` | `dict \| list \| None` | `None` | Keys to align DataFrame rows before comparison |
| `rtol` | `float` | `0.01` | Relative tolerance for numeric comparison |
| `atol` | `float` | `1e-5` | Absolute tolerance for numeric comparison |
| `profile_funcs` | `list[callable] \| None` | `None` | Functions to attach to the line profiler |
| `dump_mismatches` | `bool` | `True` | Export mismatch details to CSV |
| `show_output_heads` | `bool` | `True` | Display aligned DataFrame previews |
| `user_name` | `str` | `"Uday"` | Name shown in dashboard greeting |
| `iterations` | `int` | `1` | Number of timed iterations |
| `warmup_runs` | `int` | `1` | Number of warmup runs (discarded) |
| `output_dir` | `str \| None` | `None` | Directory for output files |
| `history_file` | `str \| None` | `None` | JSON file for historical tracking |
| `verbose` | `bool` | `True` | Show full dashboard header |

### `BenchmarkResult`

| Field | Type | Description |
|---|---|---|
| `new_output` | `Any` | Output from the optimized function |
| `old_time_stats` | `float \| dict` | Baseline timing (or `{min, mean, median, stddev}`) |
| `new_time_stats` | `float \| dict` | Optimized timing |
| `old_memory_stats` | `float \| dict` | Baseline peak memory in MB |
| `new_memory_stats` | `float \| dict` | Optimized peak memory in MB |
| `speedup` | `float` | Ratio of baseline to optimized median time |
| `valid` | `bool` | `True` if all DataFrames passed validation |
| `mismatch_count` | `int` | Number of DataFrames with mismatches |

## Requirements

- Python >= 3.9
- pandas
- numpy
- line-profiler
- rich

## License

MIT
