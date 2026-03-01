"""
Test suite for the shrimpbench package.

Conventions used throughout:
  - verbose=False     : suppresses the Rich dashboard header
  - output_dir=None   : prevents any file I/O during tests
  - iterations=1      : keeps tests fast and produces scalar stats
  - warmup_runs=0     : skips warmup to further reduce runtime
  - dump_mismatches=False : prevents CSV writes
  - Small DataFrames  : 3-5 rows is sufficient for all validation logic
"""

import pytest
import pandas as pd
import numpy as np

from shrimpbench import run_benchmark, BenchmarkResult
from shrimpbench.runner import _extract_return_names


# ---------------------------------------------------------------------------
# Module-level helpers for _extract_return_names tests
# (Must be at module level — getsource on nested functions yields indented
# code that ast.parse rejects as an IndentationError.)
# ---------------------------------------------------------------------------

def _func_named_tuple():
    foo = 1
    bar = 2
    return foo, bar


def _func_single_value():
    result = 42
    return result


def _func_no_return():
    x = 1  # noqa


def _func_expr_in_tuple():
    a = 1
    return a, a + 1  # 'a + 1' is a BinOp, not a Name


def _func_with_inner_closure():
    # Outer returns a single value; inner helper returns a tuple.
    # _extract_return_names must NOT pick up the inner tuple.
    def _helper():
        x, y = 1, 2
        return x, y
    result = _helper()
    return result


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({"key": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})


@pytest.fixture
def simple_scope(simple_df):
    return {"df": simple_df}


@pytest.fixture
def run_opts():
    return dict(
        verbose=False,
        output_dir=None,
        iterations=1,
        warmup_runs=0,
        dump_mismatches=False,
        show_output_heads=False,
    )


# ---------------------------------------------------------------------------
# TestBenchmarkResultDataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResultDataclass:
    def test_default_construction(self):
        result = BenchmarkResult()
        assert result.new_output is None
        assert result.old_time_stats is None
        assert result.new_time_stats is None
        assert result.old_memory_stats is None
        assert result.new_memory_stats is None
        assert result.speedup == 0.0
        assert result.valid is True
        assert result.mismatch_count == 0

    def test_explicit_construction(self):
        result = BenchmarkResult(new_output=42, speedup=2.5, valid=False, mismatch_count=3)
        assert result.new_output == 42
        assert result.speedup == 2.5
        assert result.valid is False
        assert result.mismatch_count == 3

    def test_has_expected_fields(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(BenchmarkResult)}
        expected = {
            "old_output", "new_output", "old_time_stats", "new_time_stats",
            "old_memory_stats", "new_memory_stats",
            "speedup", "valid", "mismatch_count",
        }
        assert expected == field_names


# ---------------------------------------------------------------------------
# TestExtractReturnNames
# ---------------------------------------------------------------------------

class TestExtractReturnNames:
    def test_named_tuple_return(self):
        names = _extract_return_names(_func_named_tuple)
        assert names == ["foo", "bar"]

    def test_single_value_return(self):
        assert _extract_return_names(_func_single_value) is None

    def test_no_return_statement(self):
        assert _extract_return_names(_func_no_return) is None

    def test_expression_in_tuple_gives_none_placeholder(self):
        names = _extract_return_names(_func_expr_in_tuple)
        assert names == ["a", None]

    def test_lambda_returns_none_gracefully(self):
        f = lambda x: (x, x + 1)  # noqa: E731
        result = _extract_return_names(f)
        # Must not raise; returns None or a list
        assert result is None or isinstance(result, list)

    def test_nested_function_tuple_return(self):
        # Functions defined inside other functions (e.g. inside a Jupyter cell
        # or a helper scope) produce indented source. textwrap.dedent fixes this.
        def make_func():
            def inner():
                alpha = 1
                beta = 2
                return alpha, beta
            return inner

        names = _extract_return_names(make_func())
        assert names == ["alpha", "beta"]

    def test_outer_with_nested_closure_not_confused(self):
        # Previously ast.walk visited the inner helper's FunctionDef and
        # returned its tuple names instead of the outer function's return.
        # The fix (look only at the outermost FunctionDef) prevents this.
        names = _extract_return_names(_func_with_inner_closure)
        assert names is None  # outer returns a single value, not a tuple


# ---------------------------------------------------------------------------
# TestRunBenchmarkReturnType
# ---------------------------------------------------------------------------

class TestRunBenchmarkReturnType:
    def test_returns_benchmark_result(self, simple_scope, run_opts):
        def f(df): return df
        result = run_benchmark(f, f, simple_scope, **run_opts)
        assert isinstance(result, BenchmarkResult)

    def test_all_stats_fields_populated(self, simple_scope, run_opts):
        def f(df): return df
        result = run_benchmark(f, f, simple_scope, **run_opts)
        assert result.old_time_stats is not None
        assert result.new_time_stats is not None
        assert result.old_memory_stats is not None
        assert result.new_memory_stats is not None

    def test_speedup_is_positive_float(self, simple_scope, run_opts):
        def f(df): return df
        result = run_benchmark(f, f, simple_scope, **run_opts)
        assert isinstance(result.speedup, float)
        assert result.speedup > 0

    def test_new_output_is_set(self, simple_scope, run_opts):
        def f_old(df): return df
        def f_new(df): return df.copy()
        result = run_benchmark(f_old, f_new, simple_scope, **run_opts)
        assert result.new_output is not None

    def test_old_output_in_result(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **run_opts)
        assert result.old_output is not None
        assert isinstance(result.old_output, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestValidationHappyPath
# ---------------------------------------------------------------------------

class TestValidationHappyPath:
    def test_identical_dataframes_valid(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **run_opts)
        assert result.valid is True
        assert result.mismatch_count == 0

    def test_both_empty_dataframes_valid(self, run_opts):
        scope = {"df": pd.DataFrame()}
        def f(df): return pd.DataFrame()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True
        assert result.mismatch_count == 0

    def test_numerically_close_values_valid(self, run_opts):
        df = pd.DataFrame({"key": ["a"], "value": [1.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df):
            out = df.copy()
            out["value"] = out["value"] + 1e-9  # within default atol=1e-8? No, 1e-9 < 1e-8
            return out
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is True

    def test_nan_in_numeric_column_treated_as_equal(self, run_opts):
        df = pd.DataFrame({"key": ["a", "b"], "value": [1.0, float("nan")]})
        scope = {"df": df}
        def f(df): return df.copy()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True

    def test_nan_in_string_column_treated_as_equal(self, run_opts):
        df = pd.DataFrame({"key": ["a", None], "value": [1.0, 2.0]})
        scope = {"df": df}
        def f(df): return df.copy()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True

    def test_scalar_return_is_skipped(self, run_opts):
        scope = {"x": 10}
        def f_old(x): return x * 2
        def f_new(x): return x * 3  # different value, but non-DF so skipped
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is True
        assert result.mismatch_count == 0

    def test_dict_return_is_skipped(self, run_opts):
        scope = {"x": 5}
        def f_old(x): return {"result": x}
        def f_new(x): return {"result": x + 1}
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is True


# ---------------------------------------------------------------------------
# TestValidationFailurePaths
# ---------------------------------------------------------------------------

class TestValidationFailurePaths:
    def test_value_mismatch_sets_invalid(self, run_opts):
        df = pd.DataFrame({"key": ["a", "b"], "value": [1.0, 2.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df):
            out = df.copy()
            out["value"] = out["value"] * 100
            return out
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is False
        assert result.mismatch_count > 0

    def test_row_count_mismatch_invalid(self, run_opts):
        df = pd.DataFrame({"key": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df): return df.iloc[:2].copy()
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is False

    def test_one_empty_one_nonempty_invalid(self, run_opts):
        df = pd.DataFrame({"key": ["a"], "value": [1.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df): return pd.DataFrame(columns=df.columns)
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is False

    def test_tuple_length_mismatch_invalid(self, run_opts):
        df = pd.DataFrame({"key": ["a"], "value": [1.0]})
        scope = {"df": df}
        def f_old(df): return df.copy(), df.copy()
        def f_new(df): return df.copy(), df.copy(), df.copy()
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is False

    def test_string_column_mismatch_invalid(self, run_opts):
        # Use a numeric id column as the merge key so 'label' is treated as a
        # value column (not a sort key), allowing string comparison to occur.
        df = pd.DataFrame({"id": [1, 2], "label": ["cat", "dog"]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df):
            out = df.copy()
            out.loc[1, "label"] = "fish"
            return out
        result = run_benchmark(f_old, f_new, scope, merge_keys={0: ["id"]}, **run_opts)
        assert result.valid is False
        assert result.mismatch_count > 0


# ---------------------------------------------------------------------------
# TestTupleReturns
# ---------------------------------------------------------------------------

class TestTupleReturns:
    def test_matching_two_tuple_valid(self, run_opts):
        df1 = pd.DataFrame({"key": ["a"], "v": [1.0]})
        df2 = pd.DataFrame({"key": ["b"], "v": [2.0]})
        scope = {"df1": df1, "df2": df2}
        def f(df1, df2): return df1.copy(), df2.copy()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True

    def test_mismatch_in_second_tuple_element_invalid(self, run_opts):
        df1 = pd.DataFrame({"key": ["a"], "v": [1.0]})
        df2 = pd.DataFrame({"key": ["b"], "v": [2.0]})
        scope = {"df1": df1, "df2": df2}
        def f_old(df1, df2): return df1.copy(), df2.copy()
        def f_new(df1, df2):
            bad = df2.copy()
            bad["v"] = 999.0
            return df1.copy(), bad
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is False
        assert result.mismatch_count > 0

    def test_mixed_tuple_scalar_and_df(self, run_opts):
        df = pd.DataFrame({"key": ["a"], "v": [1.0]})
        scope = {"df": df}
        def f(df): return 42, df.copy()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True


# ---------------------------------------------------------------------------
# TestMergeKeys
# ---------------------------------------------------------------------------

class TestMergeKeys:
    def test_user_provided_merge_keys_dict(self, run_opts):
        # Both functions receive the same df; f_new returns rows in a different order.
        # With merge_keys telling the validator to sort by 'id', they should align.
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df): return df.iloc[::-1].reset_index(drop=True)
        result = run_benchmark(f_old, f_new, scope, merge_keys={0: ["id"]}, **run_opts)
        assert result.valid is True

    def test_auto_detect_string_columns_as_sort_keys(self, run_opts):
        # Non-numeric column 'category' is auto-detected as the sort key.
        df = pd.DataFrame({"category": ["x", "y"], "score": [1.0, 2.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df): return df.iloc[::-1].reset_index(drop=True)  # reversed rows
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is True

    def test_all_numeric_df_falls_back_to_all_columns(self, run_opts):
        # No non-numeric columns — auto-detect falls back to all columns.
        df = pd.DataFrame({"a": [3, 1, 2], "b": [30.0, 10.0, 20.0]})
        scope = {"df": df}
        def f_old(df): return df.copy()
        def f_new(df): return df.iloc[::-1].reset_index(drop=True)  # reversed rows
        result = run_benchmark(f_old, f_new, scope, **run_opts)
        assert result.valid is True

    def test_missing_merge_key_falls_back_gracefully(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(
            f, f, simple_scope,
            merge_keys={0: ["nonexistent_col"]},
            **run_opts,
        )
        assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# TestParameterAutoDiscovery
# ---------------------------------------------------------------------------

class TestParameterAutoDiscovery:
    def test_function_uses_subset_of_scope(self, run_opts):
        scope = {
            "df": pd.DataFrame({"k": ["a"], "v": [1.0]}),
            "extra_var": "ignored",
            "another_var": 99,
        }
        def f(df): return df.copy()
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True

    def test_missing_required_param_returns_none(self, run_opts):
        scope = {}
        def f(df): return df
        result = run_benchmark(f, f, scope, **run_opts)
        assert result is None

    def test_optional_param_not_in_scope_uses_default(self, run_opts):
        df = pd.DataFrame({"k": ["a"], "v": [1.0]})
        scope = {"df": df}
        def f(df, multiplier=2.0):
            out = df.copy()
            out["v"] = out["v"] * multiplier
            return out
        result = run_benchmark(f, f, scope, **run_opts)
        assert result.valid is True


# ---------------------------------------------------------------------------
# TestStatsShape
# ---------------------------------------------------------------------------

class TestStatsShape:
    def test_single_iteration_stats_are_floats(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **run_opts)  # iterations=1
        assert isinstance(result.old_time_stats, float)
        assert isinstance(result.new_time_stats, float)
        assert isinstance(result.old_memory_stats, float)
        assert isinstance(result.new_memory_stats, float)

    def test_multi_iteration_stats_are_dicts(self, simple_scope, run_opts):
        opts = {**run_opts, "iterations": 3}
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **opts)
        expected_keys = {"min", "mean", "median", "stddev"}
        for field in (result.old_time_stats, result.new_time_stats,
                      result.old_memory_stats, result.new_memory_stats):
            assert isinstance(field, dict)
            assert set(field.keys()) == expected_keys

    def test_stats_dict_values_are_non_negative(self, simple_scope, run_opts):
        opts = {**run_opts, "iterations": 3}
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **opts)
        for field in (result.old_time_stats, result.new_time_stats,
                      result.old_memory_stats, result.new_memory_stats):
            for key in ("min", "mean", "median", "stddev"):
                assert field[key] >= 0.0

    def test_warmup_zero_works(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **{**run_opts, "warmup_runs": 0})
        assert isinstance(result, BenchmarkResult)
        assert result.valid is True

    def test_warmup_nonzero_works(self, simple_scope, run_opts):
        def f(df): return df.copy()
        result = run_benchmark(f, f, simple_scope, **{**run_opts, "warmup_runs": 2})
        assert result.valid is True

    def test_profiler_does_not_contaminate_timing(self, simple_scope, run_opts):
        # With the fix, the profiling pass is separate from the timing loop.
        # All timing samples should be in the same ballpark — no outlier on
        # the last iteration from profiler overhead.
        def f(df): return df.copy()
        opts = {**run_opts, "iterations": 5, "profile_funcs": [f]}
        result = run_benchmark(f, f, simple_scope, **opts)
        assert isinstance(result, BenchmarkResult)
        # Stats should be a dict (multi-iteration)
        assert isinstance(result.old_time_stats, dict)
        # stddev should be small relative to mean — a contaminated last sample
        # would produce a stddev that dwarfs the mean (e.g. stddev/mean > 1.0)
        stats = result.old_time_stats
        if stats["mean"] > 0:
            assert stats["stddev"] / stats["mean"] < 2.0


# ---------------------------------------------------------------------------
# TestPublicAPIImports
# ---------------------------------------------------------------------------

class TestPublicAPIImports:
    def test_run_benchmark_importable(self):
        from shrimpbench import run_benchmark
        assert callable(run_benchmark)

    def test_benchmark_result_importable(self):
        from shrimpbench import BenchmarkResult
        assert BenchmarkResult is not None

    def test_extract_return_names_importable_from_runner(self):
        from shrimpbench.runner import _extract_return_names
        assert callable(_extract_return_names)
