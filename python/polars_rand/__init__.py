from math import pi
import polars as pl

from polars_rand.helpers import into_expr, IntoExpr, sample

# TODO: write docstrings


def normal(mean: IntoExpr = 0, variance: IntoExpr = 1) -> pl.Expr:
    return sample("standard_normal") * into_expr(variance) + into_expr(mean)


def uniform(low: IntoExpr = 0, high: IntoExpr = 1) -> pl.Expr:
    low, high = into_expr(low), into_expr(high)
    return low + sample("standard_uniform") * (high - low)


def bernoulli(p: IntoExpr = 0.5) -> pl.Expr:
    return sample("standard_uniform") < into_expr(p)


def binomial(p: IntoExpr, n: IntoExpr) -> pl.Expr:
    p, n = into_expr(p, pl.Float64), into_expr(n, pl.UInt64)
    return sample("binomial", p, n)


def exponential(scale: IntoExpr = 1) -> pl.Expr:
    return sample("standard_uniform").log().neg().mul(scale)


def poisson(rate: IntoExpr = 1) -> pl.Expr:
    rate = into_expr(rate, pl.Float64)
    return sample("poisson", rate)


def weibull(scale: IntoExpr = 1, shape: IntoExpr = 1):
    return exponential(scale).pow(shape)


def cauchy(
    # TODO: figure out how to sample arbitrary parameters
    # x_0: IntoExpr,
    # gamma: IntoExpr,
):
    return normal() / normal()


def laplace(mean: IntoExpr = 0, scale: IntoExpr = 1):
    u = uniform(-1, 1)
    return into_expr(mean) - into_expr(scale) * u.sign() * (1 - u.abs()).log()


def gamma(shape: IntoExpr, scale: IntoExpr):
    shape, scale = into_expr(shape, pl.Float64), into_expr(scale, pl.Float64)
    return sample("gamma", shape, scale)


def beta(alpha: IntoExpr, beta: IntoExpr):
    x = gamma(alpha, 1)
    y = gamma(beta, 1)
    return x / (x + y)
