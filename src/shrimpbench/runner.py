import time
import tracemalloc
import inspect
import ast
import warnings
import os
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from datetime import datetime
from line_profiler import LineProfiler

# --- RICH CLI IMPORTS ---
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule


@dataclass
class BenchmarkResult:
    """Structured result returned by run_benchmark."""
    new_output: Any = None
    old_time_stats: Any = None  # float or dict with min/mean/median/stddev
    new_time_stats: Any = None
    old_memory_stats: Any = None
    new_memory_stats: Any = None
    speedup: float = 0.0
    valid: bool = True
    mismatch_count: int = 0


def _extract_return_names(func):
    """Try to extract variable names from a function's return tuple via AST."""
    try:
        src = inspect.getsource(func)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for stmt in reversed(node.body):
                    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
                        names = []
                        for el in stmt.value.elts:
                            if isinstance(el, ast.Name):
                                names.append(el.id)
                            else:
                                names.append(None)
                        return names
    except (OSError, TypeError, SyntaxError):
        pass
    return None


def run_benchmark(func_old, func_new, scope,
                  alias_old="Baseline",
                  alias_new="Optimized",
                  merge_keys=None,
                  rtol=0.01,
                  atol=1e-5,
                  profile_funcs=None,
                  dump_mismatches=True,
                  show_output_heads=True,
                  user_name="Uday",
                  iterations=1,
                  warmup_runs=1,
                  output_dir=None,
                  history_file=None,
                  verbose=True):
    """
    Zero-effort Benchmarking Utility with Tuple Support,
    Multi-DataFrame Row Alignment, File Profiling, Aligned Previews, and Rich CLI Dashboard.

    Compares two implementations (old vs new) of the same logic by executing both,
    measuring execution time and peak memory usage, and then running a smart validation
    engine that checks output DataFrames for row-count mismatches, column differences,
    and value-level discrepancies (within configurable tolerance).

    Args:
        func_old (callable): The baseline/original function to benchmark against.
        func_new (callable): The optimized/new function to compare.
        scope (dict): A dictionary of variable names to values that will be auto-mapped
            as keyword arguments to both functions based on their signatures.
        alias_old (str): Display name for the baseline function in output. Defaults to "Baseline".
        alias_new (str): Display name for the optimized function in output. Defaults to "Optimized".
        merge_keys (dict | None): A dict mapping DataFrame index (int) to a column name (or list
            of column names) used to align rows before comparison. If None, rows are compared
            by position after sorting.
        rtol (float): Relative tolerance for numeric comparison. Defaults to 0.01 (1%).
        atol (float): Absolute tolerance for numeric comparison. Defaults to 1e-5.
        profile_funcs (list[callable] | None): List of functions to attach to the line profiler.
            Profiling runs only on the last iteration. Results are saved to a text file.
        dump_mismatches (bool): If True, writes detailed mismatch reports (CSV + JSON) to
            the output directory. Defaults to True.
        show_output_heads (bool): If True, displays aligned head previews of old vs new
            DataFrames side by side in the console. Defaults to True.
        user_name (str): Name displayed in the CLI dashboard greeting. Defaults to "Uday".
        iterations (int): Number of timed iterations to run per function (after warmup).
            When > 1, reports min/mean/median/stddev of timings. Defaults to 1.
        warmup_runs (int): Number of warmup executions (results discarded) before timed
            iterations begin. Helps stabilize JIT/caching effects. Defaults to 1.
        output_dir (str | None): Directory path for saving profiler output, mismatch reports,
            and benchmark history. Created automatically if it doesn't exist.
        history_file (str | None): Path to a JSON file for appending benchmark results over
            time, enabling historical trend tracking.
        verbose (bool): If True, displays the full ASCII dashboard header with tips and
            metadata. Set to False for minimal output. Defaults to True.

    Returns:
        BenchmarkResult: A dataclass containing:
            - new_output: The output from the optimized function.
            - old_time_stats / new_time_stats: Execution time (float or dict with min/mean/median/stddev).
            - old_memory_stats / new_memory_stats: Peak memory in MB (float or dict).
            - speedup (float): Ratio of old median time to new median time.
            - valid (bool): True if all DataFrame outputs passed validation.
            - mismatch_count (int): Total number of value-level mismatches found.
    """
    wall_clock_start = time.perf_counter()

    console = Console()
    theme_color = "#DE7356"

    # --- Output directory setup ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _out_path(filename):
        if output_dir:
            return os.path.join(output_dir, filename)
        return filename

    # --- Unified Tag System ---
    TAG_INFO = f"[bold white on {theme_color}] INFO [/]"
    TAG_RUN  = f"[bold white on {theme_color}] RUN  [/]"
    TAG_WARN = "[bold white on yellow] WARN [/]"
    TAG_FAIL = "[bold white on red] FAIL [/]"
    TAG_PASS = "[bold white on #2d7d46] PASS [/]"
    TAG_OUT  = f"[bold white on {theme_color}] OUT  [/]"

    # --- 1. THE DASHBOARD HEADER ---
    if verbose:
        shrimp_ascii = (
            f"[{theme_color}]    ╲╲[/{theme_color}]\n"
            f"[{theme_color}]  ▄▀[/][bold white]●[/][{theme_color}]▀▀▄[/{theme_color}]\n"
            f"[{theme_color}] █▓▓▓▓▓█[/{theme_color}]\n"
            f"[{theme_color}]  ▀▓▓▓▀[/{theme_color}]\n"
            f"[{theme_color}]   ▀▓▀[/{theme_color}]\n"
            f"[{theme_color}]   ▟▀▙[/{theme_color}]"
        )

        left_content = (
            f"[bold green]Welcome back, {user_name}![/bold green]\n\n"
            f"{shrimp_ascii}\n\n\n"
            f"[white]Target: {alias_old} vs {alias_new}[/white]\n"
            "[grey53]Status: Initializing Engines...[/grey53]"
        )

        now = datetime.now()
        iter_info = f"Iterations: {iterations} (warmup: {warmup_runs})" if iterations > 1 else "Single-run mode"
        right_content = (
            f"[bold {theme_color}]Tips for getting started[/]\n"
            "Pass a dict to [bold white]merge_keys[/bold white] to align\n"
            "different DataFrames in a tuple individually.\n\n"
            f"[bold {theme_color}]Recent activity[/]\n"
            "Aligned Output Previews enabled.\n"
            "Profiler output routed to text file.\n\n"
            f"[grey53]{iter_info}[/grey53]\n"
            f"[grey53]{now.strftime('%B %d, %Y  %I:%M %p')}[/grey53]"
        )

        dash_table = Table(show_header=False, show_edge=False, box=box.SIMPLE, padding=(1, 3), expand=True)
        dash_table.add_column(ratio=1, justify="center")
        dash_table.add_column(ratio=1, justify="left")
        dash_table.add_row(left_content, right_content)

        header_panel = Panel(
            dash_table,
            title=f"[bold {theme_color}] Shrimp Bench v2.0.0 [/]",
            subtitle="[grey53] Performance Benchmarking Toolkit [/]",
            title_align="left",
            subtitle_align="left",
            border_style=theme_color,
            box=box.ROUNDED
        )

        console.print()
        console.print(header_panel)
        console.print()

    # --- 2. AUTO-DISCOVERY ENGINE ---
    def _get_required_and_optional(func):
        params = inspect.signature(func).parameters
        required, optional = set(), set()
        for name, p in params.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is p.empty:
                required.add(name)
            else:
                optional.add(name)
        return required, optional

    req_old, opt_old = _get_required_and_optional(func_old)
    req_new, opt_new = _get_required_and_optional(func_new)
    all_required = req_old | req_new
    all_optional = (opt_old | opt_new) - all_required
    all_names = all_required | all_optional

    # Hard-stop only on truly required params
    missing_required = all_required - scope.keys()
    if missing_required:
        console.print(f"{TAG_FAIL} [bold red]Missing required parameters in scope: {', '.join(sorted(missing_required))}[/]")
        console.print("       [grey74]Hint: ensure your `scope` dict contains all variables needed by the target functions.[/grey74]")
        return None

    # Warn about missing optional params
    missing_optional = all_optional - scope.keys()
    if missing_optional:
        console.print(f"{TAG_WARN} [yellow]Optional parameters not in scope (using defaults): {', '.join(sorted(missing_optional))}[/]")

    inputs = {k: scope[k] for k in all_names if k in scope}
    console.print(f"{TAG_INFO} [grey74]Auto-mapped variables: {', '.join(inputs.keys())}[/grey74]")

    # --- 3. PROFILER SETUP ---
    lp = LineProfiler()
    if profile_funcs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for f in profile_funcs:
                inner_func = getattr(f, '__wrapped__', f)
                lp.add_function(inner_func)

    def execute_with_metrics(func, name, num_warmup, num_iterations):
        """Run warmup + timed iterations, return (output, time_list, memory_list)."""
        # Warmup runs (discarded)
        for w in range(num_warmup):
            with console.status(f"[{theme_color}]Warming up {name} (run {w+1}/{num_warmup})...", spinner="dots"):
                func(**inputs)

        time_results = []
        mem_results = []
        final_output = None

        for i in range(num_iterations):
            label = f"Running {name}" if num_iterations == 1 else f"Running {name} (iteration {i+1}/{num_iterations})"
            with console.status(f"[{theme_color}]{label}...", spinner="dots"):
                tracemalloc.start()
                t0 = time.perf_counter()

                if profile_funcs and i == num_iterations - 1:
                    # Only profile on the last iteration
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        lp_wrapper = lp(func)
                    out = lp_wrapper(**inputs)
                else:
                    out = func(**inputs)

                t1 = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                time_results.append(t1 - t0)
                mem_results.append(peak / (1024**2))
                final_output = out

        return final_output, time_results, mem_results

    def _compute_stats(values):
        """Compute min/mean/median/stddev from a list of values."""
        if len(values) == 1:
            return values[0]
        return {
            "min": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _stat_median(stats):
        """Extract median value from stats (float or dict)."""
        if isinstance(stats, dict):
            return stats["median"]
        return stats

    def _stat_mean(stats):
        if isinstance(stats, dict):
            return stats["mean"]
        return stats

    def _stat_stddev(stats):
        if isinstance(stats, dict):
            return stats["stddev"]
        return 0.0

    # --- 4. EXECUTION ---
    console.print(f"{TAG_RUN} [grey74]Running {alias_old} version...[/grey74]")
    console.print(Rule(style=theme_color))
    old_out, old_times, old_mems = execute_with_metrics(func_old, alias_old, warmup_runs, iterations)

    console.print(f"{TAG_RUN} [grey74]Running {alias_new} version...[/grey74]")
    console.print(Rule(style=theme_color))
    new_out, new_times, new_mems = execute_with_metrics(func_new, alias_new, warmup_runs, iterations)

    console.print(f"{TAG_RUN} [grey74]Smart Validation Engine[/grey74]")
    console.print(Rule(style=theme_color))

    old_time_stats = _compute_stats(old_times)
    new_time_stats = _compute_stats(new_times)
    old_mem_stats = _compute_stats(old_mems)
    new_mem_stats = _compute_stats(new_mems)

    # --- 5. SMART VALIDATION ENGINE ---
    old_list = list(old_out) if isinstance(old_out, (tuple, list)) else [old_out]
    new_list = list(new_out) if isinstance(new_out, (tuple, list)) else [new_out]

    df_names = _extract_return_names(func_old)
    if not df_names or len(df_names) != len(old_list):
        df_names = [f"Tuple Index {i}" for i in range(len(old_list))]

    overall_valid = True
    mismatch_reports = []
    validated_count = 0
    skipped_count = 0
    per_df_rows_old = {}
    per_df_rows_new = {}

    # Tuple length mismatch detection
    if len(old_list) != len(new_list):
        console.print(f"{TAG_FAIL} [bold red]Output tuple length mismatch: {alias_old} returned {len(old_list)} items, {alias_new} returned {len(new_list)} items (names extracted from {func_old.__name__})[/]")
        overall_valid = False

    for idx, (o_df, n_df) in enumerate(zip(old_list, new_list)):
        if hasattr(o_df, "to_pandas"):
            o_df = o_df.to_pandas()
        if hasattr(n_df, "to_pandas"):
            n_df = n_df.to_pandas()

        if not hasattr(o_df, 'reindex') or not hasattr(n_df, 'reindex'):
            console.print(f"{TAG_INFO} [grey74]Skipping non-DataFrame: {df_names[idx]}[/grey74]")
            skipped_count += 1
            continue

        if o_df.empty and n_df.empty:
            if show_output_heads:
                console.print(f"[grey53] {df_names[idx]} DataFrame is empty in both versions[/]")
            validated_count += 1
            continue

        per_df_rows_old[df_names[idx]] = len(o_df)
        per_df_rows_new[df_names[idx]] = len(n_df)

        # 5a. Multi-DataFrame Row Alignment Logic
        current_keys = None
        if isinstance(merge_keys, dict):
            current_keys = merge_keys.get(idx)
        elif isinstance(merge_keys, list):
            if all(isinstance(i, list) for i in merge_keys):
                current_keys = merge_keys[idx] if idx < len(merge_keys) else None
            else:
                current_keys = [c for c in merge_keys if c in o_df.columns]

        if current_keys:
            missing_keys = [k for k in current_keys if k not in o_df.columns or k not in n_df.columns]
            if missing_keys:
                console.print(f"{TAG_WARN} Missing keys {missing_keys} in {df_names[idx]}. Falling back to auto-detect.")
                current_keys = None

        if not current_keys:
            current_keys = list(o_df.select_dtypes(exclude=[np.number]).columns)
            if not current_keys:
                current_keys = list(o_df.columns)

        # Sorting
        try:
            o_df = o_df.sort_values(by=current_keys).reset_index(drop=True)
            n_df = n_df.sort_values(by=current_keys).reset_index(drop=True)
        except Exception as e:
            console.print(f"{TAG_FAIL} Sort failed on {df_names[idx]} using keys {current_keys}: {e}")
            overall_valid = False
            continue

        # --- 5b. ALIGNED OUTPUT PREVIEWS ---
        if show_output_heads:
            console.print(f"\n[bold white]{df_names[idx]}[/] [grey53]— {alias_old} vs {alias_new}[/]")
            console.print(Rule(style="grey53"))

            def _df_to_rich_table(df, title, title_color):
                preview = df.head(5)
                tbl = Table(
                    title=f"[{title_color}]{title}[/] [grey53](First 5 Rows - Sorted)[/]",
                    box=box.SIMPLE_HEAVY,
                    border_style="grey53",
                    header_style="bold white",
                    show_lines=False,
                    padding=(0, 1),
                )
                for col_name in preview.columns:
                    tbl.add_column(str(col_name), justify="right" if pd.api.types.is_numeric_dtype(preview[col_name]) else "left")
                for _, row in preview.iterrows():
                    tbl.add_row(*[str(v) for v in row.values])
                return tbl

            if not o_df.empty:
                console.print(_df_to_rich_table(o_df, f"▼ {alias_old}", theme_color))
            else:
                console.print("Empty DataFrame")

            if not n_df.empty:
                console.print(_df_to_rich_table(n_df, f"▼ {alias_new}", "green"))
            else:
                console.print("Empty DataFrame")

            console.print()

        # 5c. Row Count Verification
        if len(o_df) != len(n_df):
            overall_valid = False
            mismatch_reports.append((idx, f"Row count mismatch ({len(o_df)} vs {len(n_df)})", o_df, n_df))
            continue

        # 5d. Value Comparison
        columns_to_check = [c for c in o_df.columns if c not in current_keys]
        mismatch_mask = pd.Series(False, index=o_df.index)

        for col in columns_to_check:
            if col not in n_df.columns:
                console.print(f"{TAG_FAIL} Column missing in optimized DataFrame: {col}")
                overall_valid = False
                continue

            old_vals = o_df[col].values
            new_vals = n_df[col].values

            if pd.api.types.is_numeric_dtype(o_df[col]):
                old_numeric = pd.to_numeric(pd.Series(old_vals), errors='coerce').values
                new_numeric = pd.to_numeric(pd.Series(new_vals), errors='coerce').values
                match = np.isclose(old_numeric, new_numeric, rtol=rtol, atol=atol, equal_nan=True)
            else:
                match = (old_vals == new_vals) | (pd.isna(old_vals) & pd.isna(new_vals))

            mismatch_mask |= ~pd.Series(match, index=o_df.index)

        validated_count += 1

        if mismatch_mask.any():
            overall_valid = False
            err_count = mismatch_mask.sum()
            mismatch_reports.append((idx, f"{err_count} value mismatches", o_df[mismatch_mask].copy(), n_df[mismatch_mask].copy()))

    # --- 6. FINAL SUMMARY TABLE ---
    console.print()

    old_t_median = _stat_median(old_time_stats)
    new_t_median = _stat_median(new_time_stats)
    old_m_median = _stat_median(old_mem_stats)
    new_m_median = _stat_median(new_mem_stats)

    speedup = old_t_median / new_t_median if new_t_median > 0 else float('inf')
    speedup_color = "green" if speedup >= 1.0 else "red"
    time_color = "green" if new_t_median <= old_t_median else "red"
    mem_color = "green" if new_m_median <= old_m_median else "red"

    time_delta = old_t_median - new_t_median
    mem_delta = old_m_median - new_m_median
    time_pct = ((new_t_median - old_t_median) / old_t_median * 100) if old_t_median > 0 else 0.0
    mem_pct = ((new_m_median - old_m_median) / old_m_median * 100) if old_m_median > 0 else 0.0

    res_table = Table(title="Benchmark Results", box=box.ROUNDED, border_style="grey53", header_style="bold white")
    res_table.add_column("Metric", justify="left")
    res_table.add_column(alias_old, justify="right", style="white")
    res_table.add_column(f"[green]{alias_new}[/green]", justify="right")
    res_table.add_column("Delta", justify="right", style="grey74")
    res_table.add_column("% Change", justify="right", style="grey74")

    if iterations > 1:
        old_t_str = f"{_stat_mean(old_time_stats):.4f} ± {_stat_stddev(old_time_stats):.4f}"
        new_t_str = f"{_stat_mean(new_time_stats):.4f} ± {_stat_stddev(new_time_stats):.4f}"
        old_m_str = f"{_stat_mean(old_mem_stats):.2f} ± {_stat_stddev(old_mem_stats):.2f}"
        new_m_str = f"{_stat_mean(new_mem_stats):.2f} ± {_stat_stddev(new_mem_stats):.2f}"
    else:
        old_t_str = f"{old_t_median:.4f}"
        new_t_str = f"{new_t_median:.4f}"
        old_m_str = f"{old_m_median:.2f}"
        new_m_str = f"{new_m_median:.2f}"

    res_table.add_row(
        "Time (s)",
        old_t_str,
        f"[{time_color}]{new_t_str}[/{time_color}]",
        f"{time_delta:+.4f}s",
        f"[{time_color}]{time_pct:+.1f}%[/{time_color}]",
    )
    res_table.add_row(
        "Memory (MB)",
        old_m_str,
        f"[{mem_color}]{new_m_str}[/{mem_color}]",
        f"{mem_delta:+.2f}MB",
        f"[{mem_color}]{mem_pct:+.1f}%[/{mem_color}]",
    )
    res_table.add_row("Speedup", "-", f"[{speedup_color}]{speedup:.2f}x[/{speedup_color}]", "-", "-")
    for name in per_df_rows_old:
        old_rows = per_df_rows_old[name]
        new_rows = per_df_rows_new.get(name, "-")
        res_table.add_row(
            f"Rows: {name}",
            str(old_rows),
            str(new_rows),
            "-",
            "-",
        )

    console.print(res_table)

    # Profiler overhead footnote
    if profile_funcs:
        console.print("[grey53]  Note: Profiler was active — time/memory includes profiler overhead[/grey53]")

    # --- 7. RESULTS & LOGGING ---
    mismatch_count = len(mismatch_reports)

    # --- 8. EXPORT PROFILER TO FILE ---
    if profile_funcs:
        prof_filename = _out_path("profile_results.txt")
        with open(prof_filename, "w") as f:
            lp.print_stats(stream=f)

    # --- 9. EXPORT MISMATCH CSVs ---
    export_lines = []
    if not overall_valid and dump_mismatches:
        for idx, err_reason, old_err_df, new_err_df in mismatch_reports:
            safe_old = alias_old.replace(' ', '_')
            safe_new = alias_new.replace(' ', '_')
            filename_old = _out_path(f"mismatches_{df_names[idx]}_{safe_old}.csv")
            filename_new = _out_path(f"mismatches_{df_names[idx]}_{safe_new}.csv")
            old_err_df.to_csv(filename_old, index=False)
            new_err_df.to_csv(filename_new, index=False)
            export_lines.append(f"{TAG_OUT} [{theme_color}]Exported {err_reason} to '{filename_old}' & '{filename_new}'[/]")

    if profile_funcs:
        export_lines.append(f"{TAG_OUT} [bold white]Profiler saved to:[/bold white] [cyan]{prof_filename}[/cyan]")

    # --- 10. HISTORICAL TRACKING ---
    if history_file:
        record = {
            "timestamp": datetime.now().isoformat(),
            "alias_old": alias_old,
            "alias_new": alias_new,
            "old_time_median": old_t_median,
            "new_time_median": new_t_median,
            "old_memory_median": old_m_median,
            "new_memory_median": new_m_median,
            "speedup": speedup,
            "valid": overall_valid,
            "iterations": iterations,
        }
        history_path = _out_path(history_file) if output_dir else history_file
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as hf:
                    history = json.load(hf)
            except (json.JSONDecodeError, IOError):
                history = []
        history.append(record)
        with open(history_path, "w") as hf:
            json.dump(history, hf, indent=2)
        export_lines.append(f"{TAG_INFO} [grey74]Appended benchmark record to {history_path}[/grey74]")

    # --- 11. SINGLE FOOTER PANEL ---
    wall_clock_end = time.perf_counter()
    wall_elapsed = wall_clock_end - wall_clock_start

    footer_parts = []
    if overall_valid:
        if validated_count > 0:
            footer_parts.append(
                f"{TAG_PASS} [bold green]ALL DATAFRAMES MATCH ({validated_count} validated, {skipped_count} skipped)[/]\n"
                f"       [grey74]All {validated_count} DataFrames validated successfully.[/grey74]"
            )
        else:
            footer_parts.append(
                f"{TAG_WARN} [bold yellow]No DataFrames were validated ({skipped_count} skipped)[/]\n"
                f"       [grey74]All returned items were non-DataFrame types.[/grey74]"
            )
    else:
        footer_parts.append(f"{TAG_FAIL} [bold red]FAILED IN {mismatch_count} DATAFRAMES[/]")

    if export_lines:
        footer_parts.append("")
        footer_parts.extend(export_lines)

    footer_parts.append("")
    footer_parts.append(f"[bold {theme_color}]Completed in {wall_elapsed:.1f}s[/]")

    footer_border = "green" if overall_valid else "red"
    console.print()
    console.print(Panel(
        "\n".join(footer_parts),
        border_style=footer_border,
        box=box.ROUNDED,
    ))

    # --- 12. RETURN STRUCTURED RESULT ---
    return BenchmarkResult(
        new_output=new_out,
        old_time_stats=old_time_stats,
        new_time_stats=new_time_stats,
        old_memory_stats=old_mem_stats,
        new_memory_stats=new_mem_stats,
        speedup=speedup,
        valid=overall_valid,
        mismatch_count=mismatch_count,
    )
