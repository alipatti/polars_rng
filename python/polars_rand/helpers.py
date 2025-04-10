from typing import Literal, Optional, Union
import polars as pl
from pathlib import Path
from polars._typing import PolarsDataType

LIB = Path(__file__).parent

IntoExpr = Union[pl.Expr, str, int, float]

# we must pass at least one expression to the plugin so that
# the rust code knows length of data it should generate
# PERF: generating this expression and then immediately binning it is very wasteful
DUMMY_EXPR = pl.int_range(pl.len())


def into_expr(expr: IntoExpr, dtype: Optional[PolarsDataType] = None) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)

    if isinstance(expr, (int, float)):
        expr = pl.lit(expr)

    return expr if dtype is None else expr.cast(dtype)


def sample(
    # distributions implemented in rust
    distribution: Literal[
        "standard_normal", "standard_uniform", "binomial", "gamma", "poisson"
    ],
    *expressions: IntoExpr,
) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=LIB,
        args=[*expressions, DUMMY_EXPR],
        kwargs=dict(distribution=distribution),
        function_name="sample",
        is_elementwise=True,
    ).alias("sample")
