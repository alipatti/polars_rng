from typing import Literal, Optional, Union
import polars as pl
from pathlib import Path
from polars._typing import PolarsDataType
from polars import selectors as cs

LIB = Path(__file__).parent

IntoExpr = Union[pl.Expr, str, int, float]
RustDist = Literal[
    "standard_normal",
    "standard_uniform",
    "binomial",
    "gamma",
    "poisson",
    "beta",
]

# we must pass at least one expression to the plugin so that
# the rust code knows length of data it should generate
DUMMY_EXPR = cs.first()


def into_expr(expr: IntoExpr, dtype: Optional[PolarsDataType] = None) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)

    if isinstance(expr, (int, float)):
        expr = pl.repeat(expr, pl.len())

    return expr if dtype is None else expr.cast(dtype)


def sample(
    # distributions implemented in rust
    distribution: RustDist,
    *expressions: IntoExpr,
) -> pl.Expr:
    return pl.plugins.register_plugin_function(
        plugin_path=LIB,
        args=expressions or DUMMY_EXPR,
        kwargs=dict(distribution=distribution),
        function_name="sample",
        is_elementwise=True,
    ).alias("sample")
